from pathlib import Path

import cv2
import numpy as np

from tryon.garment import GarmentSet
from tryon.pose import PoseResult
from tryon.warp import (
    build_correspondences,
    fit_similarity,
    warp_garment,
)


def make_synthetic_pose() -> PoseResult:
    """
    Create a deterministic synthetic body pose.

    The coordinates are deliberately simple so the warp tests
    geometry rather than MediaPipe itself.
    """

    points = {
        "left_shoulder": np.array([320.0, 180.0]),
        "right_shoulder": np.array([520.0, 180.0]),
        "left_elbow": np.array([280.0, 300.0]),
        "right_elbow": np.array([560.0, 300.0]),
        "left_hip": np.array([350.0, 500.0]),
        "right_hip": np.array([490.0, 500.0]),
        "neck": np.array([420.0, 150.0]),
    }

    visibility = {
        name: 1.0
        for name in points
    }

    return PoseResult(
    px=points,
    world={},
    visibility=visibility,
    segmentation=None,
    frame_size=(800, 800),
)


def load_synthetic_tee() -> object:
    """
    Load the existing synthetic tee from the project garments folder.
    """

    project_root = Path(__file__).resolve().parents[2]
    garments_root = project_root / "garments"

    garment_root = garments_root / "synthetic_tee"

    garment_set = GarmentSet.load(
        str(garment_root)
    )

    assert garment_set.angles, (
        "synthetic_tee contains no garment angles"
    )

    # Prefer the front view.
    if 0 in garment_set.angles:
        return garment_set.angles[0]

    # Otherwise use the first available angle.
    first_angle = sorted(
        garment_set.angles.keys()
    )[0]

    return garment_set.angles[first_angle]


def test_build_correspondences():
    """
    Verify that garment anchors correctly match visible body landmarks.
    """

    angle = load_synthetic_tee()
    pose = make_synthetic_pose()

    result = build_correspondences(
        angle,
        pose,
    )

    assert result is not None

    assert len(result) >= 2

    assert result.src.shape[1] == 2
    assert result.dst.shape[1] == 2

    assert result.src.shape[0] == len(result)
    assert result.dst.shape[0] == len(result)


def test_fit_similarity():
    """
    Verify that a similarity transformation can be estimated
    from the synthetic correspondences.
    """

    angle = load_synthetic_tee()
    pose = make_synthetic_pose()

    corr = build_correspondences(
        angle,
        pose,
    )

    assert corr is not None

    matrix = fit_similarity(
        corr
    )

    assert matrix is not None

    assert matrix.shape == (2, 3)

    assert np.all(
        np.isfinite(matrix)
    )


def test_warp_garment_returns_result():
    """
    Verify that the complete garment warp pipeline produces
    a frame-sized BGRA image.
    """

    angle = load_synthetic_tee()
    pose = make_synthetic_pose()

    result = warp_garment(
        angle,
        pose,
        frame_size=(800, 800),
        use_tps=False,
    )

    assert result is not None

    assert result.image.shape[:2] == (
        800,
        800,
    )

    assert result.image.shape[2] == 4

    assert result.affine.shape == (
        2,
        3,
    )

    assert result.n_points >= 2

    assert np.isfinite(
        result.residual_px
    )

    assert result.used_tps is False


def test_warp_garment_with_tps():
    """
    Verify that the optional TPS refinement can run successfully.
    """

    angle = load_synthetic_tee()
    pose = make_synthetic_pose()

    result = warp_garment(
        angle,
        pose,
        frame_size=(800, 800),
        use_tps=True,
    )

    assert result is not None

    assert result.image.shape[:2] == (
        800,
        800,
    )

    assert result.image.shape[2] == 4

    assert np.all(
        np.isfinite(result.image)
    )

    assert np.isfinite(
        result.residual_px
    )


def test_warp_fails_with_too_few_landmarks():
    """
    Verify that the warp safely returns None when there
    are not enough visible landmarks.
    """

    angle = load_synthetic_tee()

    pose = make_synthetic_pose()

# Create a new PoseResult with no usable visible landmarks.
    pose = PoseResult(
    px=pose.px,
    world=pose.world,
    visibility={
        name: 0.0
        for name in pose.px
    },
    segmentation=pose.segmentation,
    frame_size=pose.frame_size,
   )

    pose.visibility["neck"] = 1.0

    result = warp_garment(
        angle,
        pose,
        frame_size=(800, 800),
        use_tps=False,
    )

    assert result is None