from pathlib import Path

import numpy as np

from tryon.correspondence import (
    build_correspondence,
)
from tryon.garment_mesh import GarmentMesh
from tryon.mesh_registry import GarmentMeshRegistry
from tryon.pose import PoseResult


def make_synthetic_pose() -> PoseResult:
    """
    Create a deterministic body pose for testing.

    The coordinates intentionally form a simple upright person.
    """

    px = {
        "left_shoulder": np.array(
            [320.0, 180.0],
            dtype=np.float64,
        ),

        "right_shoulder": np.array(
            [520.0, 180.0],
            dtype=np.float64,
        ),

        "left_elbow": np.array(
            [280.0, 320.0],
            dtype=np.float64,
        ),

        "right_elbow": np.array(
            [560.0, 320.0],
            dtype=np.float64,
        ),

        "left_hip": np.array(
            [350.0, 580.0],
            dtype=np.float64,
        ),

        "right_hip": np.array(
            [490.0, 580.0],
            dtype=np.float64,
        ),
    }

    visibility = {
        name: 1.0
        for name in px
    }

    world = {}

    return PoseResult(
        px=px,
        world=world,
        visibility=visibility,
        segmentation=None,
        frame_size=(800, 800),
    )


def test_synthetic_tee_correspondence():

    project_root = Path(
        __file__
    ).resolve().parents[2]

    garments_root = (
        project_root
        / "garments"
    )

    registry = GarmentMeshRegistry(
        garments_root
    )

    mesh = registry.get(
        "synthetic_tee",
        0,
    )

    pose = make_synthetic_pose()

    result = build_correspondence(
        pose,
        mesh,
    )

    assert result.valid

    assert len(
        result.names
    ) == 7

    assert set(
        result.names
    ) == {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "neck",
    }

    assert result.source_points.shape == (
        7,
        2,
    )

    assert result.destination_points.shape == (
        7,
        2,
    )

    assert result.confidence == 1.0

    assert result.scale_x is not None

    assert result.scale_y is not None


if __name__ == "__main__":
    test_synthetic_tee_correspondence()

    print(
        "Correspondence test passed."
    )