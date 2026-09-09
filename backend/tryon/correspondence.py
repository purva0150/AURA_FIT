"""
Body ↔ garment correspondence.

This module creates the geometric correspondence between:

    MediaPipe body landmarks
                and
    garment semantic anchors.

It does NOT deform the garment.

It does NOT render the garment.

It does NOT perform TPS.

Its responsibility is only:

    PoseResult
        +
    GarmentMesh
        ↓
    CorrespondenceResult

The resulting correspondence contains:

    - source garment anchor positions
    - destination body positions
    - visibility/confidence
    - body scale information
    - garment scale information

This becomes the input to the cloth deformation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .garment_mesh import GarmentMesh
from .pose import GARMENT_LANDMARKS, PoseResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VISIBILITY_THRESHOLD = 0.5

# A garment anchor should not be moved if the corresponding body landmark
# is extremely uncertain.
MIN_REQUIRED_ANCHORS = 3

# Small epsilon used to prevent division by zero.
EPS = 1e-8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorCorrespondence:
    """
    One garment-anchor ↔ body-landmark correspondence.
    """

    name: str

    garment_vertex_index: int

    garment_xy: np.ndarray

    body_xy: np.ndarray

    visibility: float

    valid: bool


@dataclass(frozen=True)
class CorrespondenceResult:
    """
    Complete body ↔ garment correspondence for one frame.
    """

    garment_id: str

    angle_deg: int

    correspondences: Dict[str, AnchorCorrespondence]

    source_points: np.ndarray

    destination_points: np.ndarray

    names: Tuple[str, ...]

    torso_height_px: Optional[float]

    shoulder_width_px: Optional[float]

    garment_torso_height_px: Optional[float]

    garment_shoulder_width_px: Optional[float]

    scale_y: Optional[float]

    scale_x: Optional[float]

    confidence: float

    valid: bool


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Euclidean distance between two 2D points.
    """

    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


def midpoint(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Midpoint between two 2D points.
    """

    return (
        np.asarray(a, dtype=np.float64)
        + np.asarray(b, dtype=np.float64)
    ) / 2.0


# ---------------------------------------------------------------------------
# Garment geometry
# ---------------------------------------------------------------------------

def garment_anchor_points(
    mesh: GarmentMesh,
) -> Dict[str, np.ndarray]:
    """
    Extract semantic anchor coordinates from a GarmentMesh.

    The mesh already stores:

        anchor name → vertex index

    and every vertex stores its original garment-space xy position.
    """

    result: Dict[str, np.ndarray] = {}

    for name, vertex_index in mesh.anchors.items():

        vertex = mesh.vertices[vertex_index]

        result[name] = np.asarray(
            vertex.xy,
            dtype=np.float64,
        )

    return result


def garment_torso_height(
    points: Dict[str, np.ndarray],
) -> Optional[float]:
    """
    Distance between garment shoulder midpoint and hip midpoint.
    """

    required = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )

    if not all(
        name in points
        for name in required
    ):
        return None

    shoulder_mid = midpoint(
        points["left_shoulder"],
        points["right_shoulder"],
    )

    hip_mid = midpoint(
        points["left_hip"],
        points["right_hip"],
    )

    return distance(
        shoulder_mid,
        hip_mid,
    )


def garment_shoulder_width(
    points: Dict[str, np.ndarray],
) -> Optional[float]:
    """
    Distance between garment shoulder anchors.
    """

    if (
        "left_shoulder" not in points
        or "right_shoulder" not in points
    ):
        return None

    return distance(
        points["left_shoulder"],
        points["right_shoulder"],
    )


# ---------------------------------------------------------------------------
# Body geometry
# ---------------------------------------------------------------------------

def body_anchor_points(
    pose: PoseResult,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """
    Extract the garment-relevant body landmarks from PoseResult.

    `neck` is intentionally obtained through PoseResult.point(), so it uses
    the project's existing shoulder-midpoint definition.
    """

    result: Dict[str, np.ndarray] = {}

    for name in GARMENT_LANDMARKS:

        point = pose.point(name)

        if point is None:
            continue

        if pose.vis(name) < visibility_threshold:
            continue

        result[name] = np.asarray(
            point,
            dtype=np.float64,
        )

    return result


def body_torso_height(
    pose: PoseResult,
) -> Optional[float]:
    """
    Return the body's shoulder-midpoint → hip-midpoint distance.
    """

    return pose.torso_height_px()


def body_shoulder_width(
    pose: PoseResult,
) -> Optional[float]:
    """
    Return the body's shoulder width.
    """

    return pose.shoulder_width_px()


# ---------------------------------------------------------------------------
# Anchor validation
# ---------------------------------------------------------------------------

def valid_anchor_names(
    garment_points: Dict[str, np.ndarray],
    body_points: Dict[str, np.ndarray],
) -> List[str]:
    """
    Return anchor names available in both spaces.
    """

    return [
        name
        for name in GARMENT_LANDMARKS
        if name in garment_points
        and name in body_points
    ]


def correspondence_confidence(
    pose: PoseResult,
    names: List[str],
) -> float:
    """
    Mean landmark visibility over the anchors used by the correspondence.
    """

    if not names:
        return 0.0

    values = [
        pose.vis(name)
        for name in names
    ]

    return float(
        np.mean(values)
    )


# ---------------------------------------------------------------------------
# Main correspondence builder
# ---------------------------------------------------------------------------

def build_correspondence(
    pose: PoseResult,
    mesh: GarmentMesh,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    min_required_anchors: int = MIN_REQUIRED_ANCHORS,
) -> CorrespondenceResult:
    """
    Build garment-anchor → body-landmark correspondence.

    Important:
        This function does NOT modify the garment mesh.

    It only creates the point pairs required by the deformation stage.

    Example:

        garment left_shoulder
                ↓
        body left_shoulder

        garment right_shoulder
                ↓
        body right_shoulder

        ...

    Returns
    -------
    CorrespondenceResult
    """

    garment_points = garment_anchor_points(
        mesh
    )

    body_points = body_anchor_points(
        pose,
        visibility_threshold,
    )

    names = valid_anchor_names(
        garment_points,
        body_points,
    )

    # ---------------------------------------------------------------
    # Build individual correspondences
    # ---------------------------------------------------------------

    correspondences: Dict[
        str,
        AnchorCorrespondence,
    ] = {}

    for name in names:

        vertex_index = mesh.anchors[name]

        garment_xy = garment_points[name]

        body_xy = body_points[name]

        visibility = pose.vis(
            name
        )

        correspondences[name] = AnchorCorrespondence(
            name=name,

            garment_vertex_index=vertex_index,

            garment_xy=garment_xy.copy(),

            body_xy=body_xy.copy(),

            visibility=visibility,

            valid=visibility >= visibility_threshold,
        )

    # ---------------------------------------------------------------
    # Stable ordering
    # ---------------------------------------------------------------

    ordered_names = tuple(
        name
        for name in GARMENT_LANDMARKS
        if name in correspondences
    )

    # ---------------------------------------------------------------
    # Source and destination arrays
    # ---------------------------------------------------------------

    source_points = np.asarray(
        [
            correspondences[name].garment_xy
            for name in ordered_names
        ],
        dtype=np.float64,
    )

    destination_points = np.asarray(
        [
            correspondences[name].body_xy
            for name in ordered_names
        ],
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # Body scale
    # ---------------------------------------------------------------

    torso_height = body_torso_height(
        pose
    )

    shoulder_width = body_shoulder_width(
        pose
    )

    # ---------------------------------------------------------------
    # Garment scale
    # ---------------------------------------------------------------

    garment_torso = garment_torso_height(
        garment_points
    )

    garment_shoulders = garment_shoulder_width(
        garment_points
    )

    # ---------------------------------------------------------------
    # Scale ratios
    # ---------------------------------------------------------------

    scale_y = None

    if (
        torso_height is not None
        and garment_torso is not None
        and garment_torso > EPS
    ):
        scale_y = (
            torso_height
            / garment_torso
        )

    scale_x = None

    if (
        shoulder_width is not None
        and garment_shoulders is not None
        and garment_shoulders > EPS
    ):
        scale_x = (
            shoulder_width
            / garment_shoulders
        )

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    confidence = correspondence_confidence(
        pose,
        list(ordered_names),
    )

    # ---------------------------------------------------------------
    # Validity
    # ---------------------------------------------------------------

    valid = (
        len(ordered_names)
        >= min_required_anchors
    )

    return CorrespondenceResult(
        garment_id=mesh.garment_id,

        angle_deg=mesh.angle_deg,

        correspondences=correspondences,

        source_points=source_points,

        destination_points=destination_points,

        names=ordered_names,

        torso_height_px=torso_height,

        shoulder_width_px=shoulder_width,

        garment_torso_height_px=garment_torso,

        garment_shoulder_width_px=garment_shoulders,

        scale_y=scale_y,

        scale_x=scale_x,

        confidence=confidence,

        valid=valid,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_correspondence_pairs(
    result: CorrespondenceResult,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return correspondence pairs in the form:

        (garment_point, body_point)

    Useful for deformation algorithms.
    """

    return [
        (
            result.source_points[i],
            result.destination_points[i],
        )
        for i in range(
            len(result.names)
        )
    ]


def get_valid_anchor_names(
    result: CorrespondenceResult,
) -> Tuple[str, ...]:
    """
    Return the anchors participating in this correspondence.
    """

    return result.names


def describe_correspondence(
    result: CorrespondenceResult,
) -> str:
    """
    Human-readable diagnostic description.
    """

    lines = [
        "Garment ↔ body correspondence",
        f"garment: {result.garment_id}",
        f"angle: {result.angle_deg}°",
        f"valid: {result.valid}",
        f"confidence: {result.confidence:.3f}",
        f"anchors: {len(result.names)}",
    ]

    if result.scale_x is not None:
        lines.append(
            f"horizontal scale: {result.scale_x:.4f}"
        )

    if result.scale_y is not None:
        lines.append(
            f"vertical scale: {result.scale_y:.4f}"
        )

    for name in result.names:

        pair = result.correspondences[name]

        lines.append(
            f"  {name}: "
            f"garment={pair.garment_xy.tolist()} "
            f"body={pair.body_xy.tolist()} "
            f"visibility={pair.visibility:.3f}"
        )

    return "\n".join(lines)