"""
Garment mesh representation.

A garment is represented as a 2D textured mesh.

Why this exists:
    The existing project uses sparse garment anchors for similarity/TPS warping.
    That is useful, but it does not give us explicit control over how different
    parts of a garment deform.

    This module introduces a mesh representation that will later allow us to:

        garment image
              ↓
        semantic anchors
              ↓
        mesh vertices
              ↓
        body correspondence
              ↓
        cloth deformation
              ↓
        rendered garment

The mesh itself is intentionally independent of MediaPipe, OpenCV warping,
and rendering. Those responsibilities belong to other modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basic vertex
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GarmentVertex:
    """
    One vertex in garment image space.

    Attributes
    ----------
    index:
        Unique integer index used by triangle definitions.

    name:
        Human-readable name.

    xy:
        Pixel coordinate in the original garment image.

    uv:
        Normalised texture coordinate in [0, 1].
        u = horizontal coordinate
        v = vertical coordinate
    """

    index: int
    name: str
    xy: Tuple[float, float]
    uv: Tuple[float, float]


# ---------------------------------------------------------------------------
# Garment mesh
# ---------------------------------------------------------------------------

@dataclass
class GarmentMesh:
    """
    2D triangular mesh describing a garment.

    vertices:
        List of garment vertices.

    triangles:
        Integer array of shape (N, 3). Each row contains three vertex indices.

    anchors:
        Semantic body/garment correspondence.

        Example:

            {
                "neck": 0,
                "left_shoulder": 1,
                "right_shoulder": 2
            }

        The value is the index of the corresponding mesh vertex.
    """

    vertices: List[GarmentVertex]
    triangles: np.ndarray
    anchors: Dict[str, int]

# Metadata identifying the garment/view this mesh belongs to.
    garment_id: str = ""
    angle_deg: int = 0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "GarmentMesh":
        """
        Load a mesh from a JSON file.

        Expected structure:

        {
            "vertices": [
                {
                    "index": 0,
                    "name": "neck",
                    "xy": [256, 70],
                    "uv": [0.50, 0.10]
                }
            ],

            "triangles": [
                [0, 1, 2]
            ],

            "anchors": {
                "neck": 0
            }
        }
        """

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if "vertices" not in data:
            raise ValueError(
                f"Mesh file is missing 'vertices': {path}"
            )

        if "triangles" not in data:
            raise ValueError(
                f"Mesh file is missing 'triangles': {path}"
            )

        if "anchors" not in data:
            raise ValueError(
                f"Mesh file is missing 'anchors': {path}"
            )

        vertices: List[GarmentVertex] = []

        for item in data["vertices"]:

            if "index" not in item:
                raise ValueError(
                    "Every garment vertex must have an 'index'."
                )

            if "xy" not in item or len(item["xy"]) != 2:
                raise ValueError(
                    f"Invalid xy coordinate for vertex {item['index']}."
                )

            if "uv" not in item or len(item["uv"]) != 2:
                raise ValueError(
                    f"Invalid uv coordinate for vertex {item['index']}."
                )

            vertices.append(
                GarmentVertex(
                    index=int(item["index"]),
                    name=str(
                        item.get(
                            "name",
                            f"v{item['index']}"
                        )
                    ),
                    xy=(
                        float(item["xy"][0]),
                        float(item["xy"][1]),
                    ),
                    uv=(
                        float(item["uv"][0]),
                        float(item["uv"][1]),
                    ),
                )
            )

        triangles = np.asarray(
            data["triangles"],
            dtype=np.int32,
        )

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                "Garment triangles must have shape (N, 3)."
            )

        anchors = {
            str(name): int(index)
            for name, index in data["anchors"].items()
        }

        mesh = cls(
            vertices=vertices,
            triangles=triangles,
            anchors=anchors,
        )

        mesh.validate()

        return mesh

    # ------------------------------------------------------------------
    # Array helpers
    # ------------------------------------------------------------------

    @property
    def points(self) -> np.ndarray:
        """
        Return all vertex positions as an (N, 2) float64 array.
        """

        if not self.vertices:
            return np.empty(
                (0, 2),
                dtype=np.float64,
            )

        return np.asarray(
            [vertex.xy for vertex in self.vertices],
            dtype=np.float64,
        )

    @property
    def uv(self) -> np.ndarray:
        """
        Return all UV coordinates as an (N, 2) float64 array.
        """

        if not self.vertices:
            return np.empty(
                (0, 2),
                dtype=np.float64,
            )

        return np.asarray(
            [vertex.uv for vertex in self.vertices],
            dtype=np.float64,
        )

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""

        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        """Number of triangles in the mesh."""

        return len(self.triangles)

    # ------------------------------------------------------------------
    # Anchor helpers
    # ------------------------------------------------------------------

    def anchor_index(self, name: str) -> Optional[int]:
        """
        Return the vertex index associated with a semantic anchor.

        Returns None when the anchor does not exist.
        """

        return self.anchors.get(name)

    def anchor_point(self, name: str) -> Optional[np.ndarray]:
        """
        Return the original garment-space coordinate of an anchor.
        """

        index = self.anchor_index(name)

        if index is None:
            return None

        return self.points[index]

    def anchor_points(self) -> Dict[str, np.ndarray]:
        """
        Return all semantic anchors as garment-space coordinates.
        """

        points = self.points

        result: Dict[str, np.ndarray] = {}

        for name, index in self.anchors.items():
            result[name] = points[index].copy()

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate mesh consistency.

        This catches malformed dataset files early instead of allowing
        OpenCV/Numpy failures much later inside the warping pipeline.
        """

        if not self.vertices:
            raise ValueError(
                "Garment mesh contains no vertices."
            )

        vertex_indices = {
            vertex.index
            for vertex in self.vertices
        }

        if len(vertex_indices) != len(self.vertices):
            raise ValueError(
                "Garment mesh contains duplicate vertex indices."
            )

        # Make sure triangle references are valid.
        for triangle in self.triangles:

            for index in triangle:

                if int(index) not in vertex_indices:
                    raise ValueError(
                        f"Triangle references unknown vertex index: {index}"
                    )

        # Make sure anchors reference valid vertices.
        for name, index in self.anchors.items():

            if index not in vertex_indices:
                raise ValueError(
                    f"Anchor '{name}' references unknown vertex {index}"
                )

        # UV sanity check.
        uv = self.uv

        if np.any(uv < -1e-6) or np.any(uv > 1.000001):
            raise ValueError(
                "Garment UV coordinates must normally be inside [0, 1]."
            )

    # ------------------------------------------------------------------
    # Debug information
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Human-readable mesh summary useful while building the dataset.
        """

        anchor_names = ", ".join(
            sorted(self.anchors.keys())
        )

        return (
            f"GarmentMesh("
            f"vertices={self.vertex_count}, "
            f"triangles={self.triangle_count}, "
            f"anchors=[{anchor_names}]"
            f")"
        )