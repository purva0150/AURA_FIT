"""
Generate a dense triangular mesh for an existing garment asset.

The current project already contains:
    - garment PNGs
    - alpha masks
    - semantic anchor JSON files

This tool converts those sparse garment assets into a mesh representation.

For every garment angle it creates:

    image
      +
    alpha mask
      +
    semantic anchors
      +
    interior sample points
      +
    boundary points
      ↓
    Delaunay triangulation
      ↓
    filtered garment mesh
      ↓
    *_mesh.json

The generated mesh is intended for the later cloth-warping stage.

Run from the repository root:

    python backend/tools/make_garment_mesh.py

Or for a different garment:

    python backend/tools/make_garment_mesh.py --id my_tee
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

TOOLS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(TOOLS_DIR)

REPO_ROOT = os.path.dirname(BACKEND_DIR)

GARMENTS_DIR = os.path.join(
    REPO_ROOT,
    "garments",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Distance between interior grid samples.
#
# Smaller value:
#     denser mesh
#     more deformation control
#     more computation
#
# Larger value:
#     lighter mesh
#     less deformation control
#
# 32 is a good first value for the current 512x700 synthetic garment.
GRID_SPACING = 32


# Approximate spacing between points sampled from the silhouette.
BOUNDARY_SPACING = 20


# Alpha threshold used to decide whether a pixel belongs to the garment.
ALPHA_THRESHOLD = 10


# A triangle is accepted only when almost all of it lies inside the
# garment silhouette.
TRIANGLE_MASK_COVERAGE = 0.97


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_json(path: str) -> dict:
    """Load a UTF-8 JSON file."""

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str, payload: dict) -> None:
    """Write formatted JSON."""

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            payload,
            fh,
            indent=2,
        )


def normalise_uv(
    x: float,
    y: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """
    Convert image coordinates into normalized UV coordinates.

    u = horizontal position
    v = vertical position
    """

    u = x / max(width - 1, 1)
    v = y / max(height - 1, 1)

    return (
        float(np.clip(u, 0.0, 1.0)),
        float(np.clip(v, 0.0, 1.0)),
    )


def point_key(
    point: Sequence[float],
    decimals: int = 2,
) -> Tuple[float, float]:
    """
    Stable key for removing duplicate mesh points.
    """

    return (
        round(float(point[0]), decimals),
        round(float(point[1]), decimals),
    )


def inside_mask(
    mask: np.ndarray,
    point: Sequence[float],
) -> bool:
    """
    Check whether a point lies inside the garment alpha mask.
    """

    x = int(round(float(point[0])))
    y = int(round(float(point[1])))

    height, width = mask.shape[:2]

    if x < 0 or x >= width:
        return False

    if y < 0 or y >= height:
        return False

    return bool(mask[y, x] >= ALPHA_THRESHOLD)


# ---------------------------------------------------------------------------
# Point collection
# ---------------------------------------------------------------------------


def add_point(
    points: List[Tuple[float, float]],
    seen: set,
    point: Sequence[float],
) -> None:
    """
    Add a point if it has not already been added.
    """

    key = point_key(point)

    if key in seen:
        return

    seen.add(key)

    points.append(
        (
            float(point[0]),
            float(point[1]),
        )
    )


def add_anchor_points(
    points: List[Tuple[float, float]],
    seen: set,
    anchors: Dict[str, Sequence[float]],
    mask: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """
    Add semantic garment anchors.

    Returns a cleaned copy of the anchor coordinates.

    Anchors are extremely important because they later provide the
    correspondence between:

        garment space
             ↕
        body space
    """

    cleaned = {}

    for name, value in anchors.items():

        if not isinstance(value, (list, tuple)):
            continue

        if len(value) != 2:
            continue

        point = (
            float(value[0]),
            float(value[1]),
        )

        # The synthetic generator guarantees that these points correspond
        # to the garment geometry. We still perform a sanity check so that
        # malformed external assets fail gracefully.
        if not inside_mask(mask, point):
            print(
                f"Warning: anchor '{name}' is outside the garment mask."
            )

        add_point(
            points,
            seen,
            point,
        )

        cleaned[name] = point

    return cleaned


def add_interior_grid_points(
    points: List[Tuple[float, float]],
    seen: set,
    mask: np.ndarray,
    spacing: int,
) -> None:
    """
    Add regularly spaced interior points.

    These points give the cloth deformation algorithm control over the
    interior of the garment instead of deforming only the outer anchors.
    """

    height, width = mask.shape[:2]

    # Keep the grid aligned across all views.
    for y in range(
        spacing // 2,
        height,
        spacing,
    ):

        for x in range(
            spacing // 2,
            width,
            spacing,
        ):

            if mask[y, x] < ALPHA_THRESHOLD:
                continue

            add_point(
                points,
                seen,
                (x, y),
            )


def _contour_spacing(
    contour: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """
    Resample contour approximately according to arc length.

    OpenCV contours can contain a very uneven number of points. We want
    deterministic boundary sampling so the generated mesh is stable.
    """

    contour = contour.reshape(-1, 2).astype(
        np.float64
    )

    if len(contour) < 2:
        return contour

    segments = np.linalg.norm(
        np.diff(
            np.vstack(
                [contour, contour[0]]
            ),
            axis=0,
        ),
        axis=1,
    )

    perimeter = float(
        np.sum(segments)
    )

    if perimeter <= 1e-6:
        return contour[:1]

    count = max(
        3,
        int(math.ceil(perimeter / spacing)),
    )

    distances = np.linspace(
        0.0,
        perimeter,
        count,
        endpoint=False,
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(segments)]
    )

    sampled = []

    for distance in distances:

        segment_index = int(
            np.searchsorted(
                cumulative,
                distance,
                side="right",
            )
            - 1
        )

        segment_index = min(
            segment_index,
            len(segments) - 1,
        )

        segment_start = contour[
            segment_index
        ]

        segment_length = segments[
            segment_index
        ]

        if segment_length <= 1e-9:
            sampled.append(
                segment_start
            )
            continue

        local_distance = (
            distance
            - cumulative[segment_index]
        )

        t = (
            local_distance
            / segment_length
        )

        segment_end = contour[
            (segment_index + 1)
            % len(contour)
        ]

        sampled_point = (
            segment_start
            + t
            * (
                segment_end
                - segment_start
            )
        )

        sampled.append(
            sampled_point
        )

    return np.asarray(
        sampled,
        dtype=np.float64,
    )


def add_boundary_points(
    points: List[Tuple[float, float]],
    seen: set,
    mask: np.ndarray,
    spacing: float,
) -> None:
    """
    Extract the garment silhouette and add regularly spaced boundary points.
    """

    binary = np.where(
        mask >= ALPHA_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError(
            "Could not find a garment silhouette."
        )

    # The garment should normally have one dominant outer contour.
    contour = max(
        contours,
        key=cv2.contourArea,
    )

    sampled = _contour_spacing(
        contour,
        spacing,
    )

    for point in sampled:

        add_point(
            points,
            seen,
            point,
        )


# ---------------------------------------------------------------------------
# Delaunay triangulation
# ---------------------------------------------------------------------------


def create_subdiv(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> cv2.Subdiv2D:
    """
    Insert points into OpenCV's Delaunay subdivision.
    """

    # Subdiv2D needs a rectangle slightly larger than the image so boundary
    # points exactly on the edge don't cause insertion failures.
    rect = (
        0,
        0,
        max(width, 1),
        max(height, 1),
    )

    subdiv = cv2.Subdiv2D(rect)

    for point in points:

        x = float(point[0])
        y = float(point[1])

        try:
            subdiv.insert(
                (x, y)
            )
        except cv2.error:
            # Duplicate or numerically problematic points are harmless here;
            # they have already been deduplicated at our level.
            continue

    return subdiv


def nearest_point_index(
    point: Sequence[float],
    points: np.ndarray,
    max_distance: float = 1.5,
) -> Optional[int]:
    """
    Map a Delaunay output point back to our original point list.
    """

    target = np.asarray(
        point,
        dtype=np.float64,
    )

    distances = np.linalg.norm(
        points - target[None, :],
        axis=1,
    )

    index = int(
        np.argmin(distances)
    )

    if distances[index] > max_distance:
        return None

    return index


def triangle_inside_mask(
    mask: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    """
    Check how much of a triangle lies inside the garment mask.

    We rasterize the triangle and compare it with the alpha silhouette.

    This prevents Delaunay triangles from bridging across empty areas,
    especially around sleeves and the neckline.
    """

    height, width = mask.shape[:2]

    tri = np.round(
        triangle
    ).astype(np.int32)

    x, y, w, h = cv2.boundingRect(tri)

    if w <= 0 or h <= 0:
        return False

    x0 = max(x, 0)
    y0 = max(y, 0)

    x1 = min(
        x + w,
        width,
    )

    y1 = min(
        y + h,
        height,
    )

    if x1 <= x0 or y1 <= y0:
        return False

    local_triangle = (
        tri
        - np.array(
            [x0, y0],
            dtype=np.int32,
        )
    )

    triangle_mask = np.zeros(
        (
            y1 - y0,
            x1 - x0,
        ),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        triangle_mask,
        local_triangle,
        255,
    )

    garment_region = mask[
        y0:y1,
        x0:x1
    ]

    triangle_pixels = (
        triangle_mask > 0
    )

    if not np.any(triangle_pixels):
        return False

    garment_pixels = (
        garment_region[
            triangle_pixels
        ]
        >= ALPHA_THRESHOLD
    )

    coverage = float(
        garment_pixels.mean()
    )

    return (
        coverage
        >= TRIANGLE_MASK_COVERAGE
    )


def delaunay_triangles(
    points: Sequence[Sequence[float]],
    mask: np.ndarray,
) -> np.ndarray:
    """
    Generate a filtered Delaunay triangle list.
    """

    height, width = mask.shape[:2]

    point_array = np.asarray(
        points,
        dtype=np.float64,
    )

    subdiv = create_subdiv(
        points,
        width,
        height,
    )

    raw_triangles = subdiv.getTriangleList()

    triangles = []
    seen = set()

    for raw in raw_triangles:

        triangle = np.asarray(
            [
                [raw[0], raw[1]],
                [raw[2], raw[3]],
                [raw[4], raw[5]],
            ],
            dtype=np.float64,
        )

        # Reject triangles that are outside the image.
        if np.any(
            triangle[:, 0] < 0
        ):
            continue

        if np.any(
            triangle[:, 1] < 0
        ):
            continue

        if np.any(
            triangle[:, 0] >= width
        ):
            continue

        if np.any(
            triangle[:, 1] >= height
        ):
            continue

        indices = []

        valid = True

        for point in triangle:

            index = nearest_point_index(
                point,
                point_array,
            )

            if index is None:
                valid = False
                break

            indices.append(index)

        if not valid:
            continue

        # A triangle cannot contain repeated vertices.
        if len(set(indices)) != 3:
            continue

        # Make sure the actual triangle belongs to the garment silhouette.
        if not triangle_inside_mask(
            mask,
            triangle,
        ):
            continue

        # Canonical representation avoids duplicate triangles.
        key = tuple(
            sorted(indices)
        )

        if key in seen:
            continue

        seen.add(key)

        triangles.append(
            indices
        )

    if not triangles:
        raise ValueError(
            "No valid garment triangles were generated."
        )

    return np.asarray(
        triangles,
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------


def generate_mesh(
    image_path: str,
    anchor_path: str,
    output_path: str,
    grid_spacing: int = GRID_SPACING,
    boundary_spacing: float = BOUNDARY_SPACING,
) -> None:
    """
    Generate one mesh JSON from one garment image.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read garment image: {image_path}"
        )

    if image.ndim != 3 or image.shape[2] < 4:
        raise ValueError(
            "Garment image must contain an alpha channel: "
            f"{image_path}"
        )

    height, width = image.shape[:2]

    alpha = image[:, :, 3]

    anchor_payload = load_json(
        anchor_path
    )

    source_anchors = anchor_payload.get(
        "anchors",
        {},
    )

    points: List[
        Tuple[float, float]
    ] = []

    seen = set()

    # ---------------------------------------------------------------
    # 1. Semantic anchors
    # ---------------------------------------------------------------

    anchors = add_anchor_points(
        points,
        seen,
        source_anchors,
        alpha,
    )

    # ---------------------------------------------------------------
    # 2. Interior points
    # ---------------------------------------------------------------

    add_interior_grid_points(
        points,
        seen,
        alpha,
        grid_spacing,
    )

    # ---------------------------------------------------------------
    # 3. Boundary points
    # ---------------------------------------------------------------

    add_boundary_points(
        points,
        seen,
        alpha,
        boundary_spacing,
    )

    if len(points) < 3:
        raise ValueError(
            f"Not enough mesh points for {image_path}"
        )

    # ---------------------------------------------------------------
    # 4. Delaunay triangulation
    # ---------------------------------------------------------------

    triangles = delaunay_triangles(
        points,
        alpha,
    )

    # ---------------------------------------------------------------
    # 5. Create vertex objects
    # ---------------------------------------------------------------

    vertices = []

    point_to_index = {}

    for index, point in enumerate(points):

        x = float(point[0])
        y = float(point[1])

        uv = normalise_uv(
            x,
            y,
            width,
            height,
        )

        vertices.append(
            {
                "index": index,
                "name": f"v{index}",
                "xy": [
                    round(x, 4),
                    round(y, 4),
                ],
                "uv": [
                    round(uv[0], 6),
                    round(uv[1], 6),
                ],
            }
        )

        point_to_index[
            point_key(point)
        ] = index

    # ---------------------------------------------------------------
    # 6. Convert semantic anchors to vertex indices
    # ---------------------------------------------------------------

    anchor_indices = {}

    for name, point in anchors.items():

        key = point_key(point)

        index = point_to_index.get(
            key
        )

        if index is None:
            raise RuntimeError(
                f"Could not map anchor '{name}' "
                "to a mesh vertex."
            )

        anchor_indices[name] = index

    # ---------------------------------------------------------------
    # 7. Final mesh payload
    # ---------------------------------------------------------------

    payload = {
        "format": "garment_mesh",
        "version": 1,

        "garment_id": anchor_payload.get(
            "garment_id"
        ),

        "angle_deg": anchor_payload.get(
            "angle_deg"
        ),

        "view": anchor_payload.get(
            "view"
        ),

        "source": {
            "image": os.path.basename(
                image_path
            ),
            "anchors": os.path.basename(
                anchor_path
            ),
        },

        "image_size": {
            "width": width,
            "height": height,
        },

        "mesh": {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
        },

        "vertices": vertices,

        "triangles": triangles.tolist(),

        "anchors": anchor_indices,

        "mesh_parameters": {
            "grid_spacing": grid_spacing,
            "boundary_spacing": boundary_spacing,
            "alpha_threshold": ALPHA_THRESHOLD,
            "triangle_mask_coverage": TRIANGLE_MASK_COVERAGE,
        },
    }

    save_json(
        output_path,
        payload,
    )

    print(
        f"  mesh: "
        f"{os.path.basename(output_path)} "
        f"({len(vertices)} vertices, "
        f"{len(triangles)} triangles)"
    )


# ---------------------------------------------------------------------------
# Garment-level generation
# ---------------------------------------------------------------------------


def generate_garment(
    garment_id: str,
    grid_spacing: int,
    boundary_spacing: float,
) -> None:
    """
    Generate meshes for every PNG/anchor pair belonging to a garment.
    """

    garment_dir = os.path.join(
        GARMENTS_DIR,
        garment_id,
    )

    if not os.path.isdir(garment_dir):
        raise FileNotFoundError(
            f"Garment directory does not exist: "
            f"{garment_dir}"
        )

    manifest_path = os.path.join(
        garment_dir,
        "garment.json",
    )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Missing garment manifest: "
            f"{manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    angles = manifest.get(
        "angles",
        [],
    )

    if not angles:
        raise ValueError(
            f"No angle entries found in {manifest_path}"
        )

    print(
        f"Generating mesh for '{garment_id}'..."
    )

    generated = 0

    for entry in angles:

        image_name = entry.get(
            "image"
        )

        anchor_name = entry.get(
            "anchors"
        )

        if not image_name:
            print(
                "  Skipping angle without image."
            )
            continue

        if not anchor_name:
            print(
                f"  Skipping {image_name}: "
                "no anchor file."
            )
            continue

        image_path = os.path.join(
            garment_dir,
            image_name,
        )

        anchor_path = os.path.join(
            garment_dir,
            anchor_name,
        )

        if not os.path.isfile(
            image_path
        ):
            print(
                f"  Missing image: {image_path}"
            )
            continue

        if not os.path.isfile(
            anchor_path
        ):
            print(
                f"  Missing anchors: {anchor_path}"
            )
            continue

        base, _ = os.path.splitext(
            image_name
        )

        output_name = (
            f"{base}_mesh.json"
        )

        output_path = os.path.join(
            garment_dir,
            output_name,
        )

        generate_mesh(
            image_path=image_path,
            anchor_path=anchor_path,
            output_path=output_path,
            grid_spacing=grid_spacing,
            boundary_spacing=boundary_spacing,
        )

        generated += 1

    print(
        f"Generated {generated} garment meshes."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate dense cloth meshes "
            "from existing garment assets."
        )
    )

    parser.add_argument(
        "--id",
        default="synthetic_tee",
        help=(
            "Garment folder inside garments/. "
            "Default: synthetic_tee"
        ),
    )

    parser.add_argument(
        "--grid-spacing",
        type=int,
        default=GRID_SPACING,
        help=(
            "Spacing between interior mesh points. "
            "Default: 32"
        ),
    )

    parser.add_argument(
        "--boundary-spacing",
        type=float,
        default=BOUNDARY_SPACING,
        help=(
            "Approximate spacing between silhouette "
            "mesh points. Default: 20"
        ),
    )

    args = parser.parse_args()

    if args.grid_spacing < 4:
        parser.error(
            "--grid-spacing must be at least 4."
        )

    if args.boundary_spacing < 4:
        parser.error(
            "--boundary-spacing must be at least 4."
        )

    generate_garment(
        garment_id=args.id,
        grid_spacing=args.grid_spacing,
        boundary_spacing=args.boundary_spacing,
    )


if __name__ == "__main__":
    main()