"""
Garment mesh registry.

This module connects the selected garment/view angle to its corresponding
dense cloth mesh.

Current asset layout:

garments/
└── synthetic_tee/
    ├── synthetic_tee_000.png
    ├── synthetic_tee_000.json
    ├── synthetic_tee_000_mesh.json
    ├── synthetic_tee_045.png
    ├── synthetic_tee_045.json
    ├── synthetic_tee_045_mesh.json
    └── ...

The registry deliberately does NOT perform warping.

Its only responsibility is:

    garment_id + angle
            ↓
       correct mesh
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .garment_mesh import GarmentMesh


class GarmentMeshRegistry:
    """
    Loads and caches garment meshes.

    Parameters
    ----------
    garments_root:
        Root directory containing garment folders.

        Example:
            project_root / "garments"
    """

    def __init__(self, garments_root: str | Path):

        self.garments_root = Path(
            garments_root
        ).resolve()

        self._cache: Dict[str, GarmentMesh] = {}

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    def _garment_directory(
        self,
        garment_id: str,
    ) -> Path:
        """
        Return the directory for one garment.
        """

        garment_dir = (
            self.garments_root
            / garment_id
        )

        if not garment_dir.is_dir():
            raise FileNotFoundError(
                f"Garment directory does not exist: "
                f"{garment_dir}"
            )

        return garment_dir

    def _mesh_path(
        self,
        garment_id: str,
        angle_deg: float,
    ) -> Path:
        """
        Resolve the mesh JSON corresponding to a garment angle.

        Angles are normalised to the project's eight-view convention:

            0, 45, 90, 135,
            180, 225, 270, 315
        """

        garment_dir = self._garment_directory(
            garment_id
        )

        angle = self.normalize_angle(
            angle_deg
        )

        filename = (
            f"{garment_id}_{angle:03d}_mesh.json"
        )

        path = garment_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Garment mesh does not exist: {path}"
            )

        return path

    # ------------------------------------------------------------------
    # Angle handling
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_angle(
        angle_deg: float,
    ) -> int:
        """
        Normalize an arbitrary angle to the nearest supported 45-degree view.

        Examples:

            0       → 0
            12      → 0
            23      → 0
            24      → 45
            44      → 45
            91      → 90
            359     → 0
            -20     → 0
        """

        angle = float(angle_deg) % 360.0

        snapped = int(
            round(angle / 45.0) * 45
        ) % 360

        return snapped

    @staticmethod
    def supported_angles() -> List[int]:
        """
        Return all currently available garment views.
        """

        return [
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
        ]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get(
        self,
        garment_id: str,
        angle_deg: float,
    ) -> GarmentMesh:
        """
        Load and return the mesh for a garment/view.

        Meshes are cached after the first load.
        """

        angle = self.normalize_angle(
            angle_deg
        )

        cache_key = (
            f"{garment_id}:{angle:03d}"
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        mesh_path = self._mesh_path(
            garment_id,
            angle,
        )

        mesh = GarmentMesh.load(
            str(mesh_path)
        )

        mesh.garment_id = garment_id
        mesh.angle_deg = angle

        self._cache[cache_key] = mesh

        return mesh

    # ------------------------------------------------------------------
    # Optional lookup helpers
    # ------------------------------------------------------------------

    def has(
        self,
        garment_id: str,
        angle_deg: float,
    ) -> bool:
        """
        Check whether a mesh exists without loading it.
        """

        try:
            path = self._mesh_path(
                garment_id,
                angle_deg,
            )

        except FileNotFoundError:
            return False

        return path.is_file()

    def available_angles(
        self,
        garment_id: str,
    ) -> List[int]:
        """
        Return angles for which mesh files actually exist.
        """

        garment_dir = self._garment_directory(
            garment_id
        )

        available = []

        for angle in self.supported_angles():

            filename = (
                f"{garment_id}_{angle:03d}_mesh.json"
            )

            if (
                garment_dir / filename
            ).is_file():

                available.append(angle)

        return available

    def clear_cache(self) -> None:
        """
        Clear all loaded meshes.
        """

        self._cache.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(
        self,
        garment_id: str,
        angle_deg: float,
    ) -> str:
        """
        Return a concise description of a selected mesh.
        """

        mesh = self.get(
            garment_id,
            angle_deg,
        )

        angle = self.normalize_angle(
            angle_deg
        )

        return (
            f"{garment_id} "
            f"@ {angle}° → "
            f"{mesh.vertex_count} vertices, "
            f"{mesh.triangle_count} triangles"
        )