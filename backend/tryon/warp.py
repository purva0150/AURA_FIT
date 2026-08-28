"""Garment warping: correspondences -> similarity fit -> thin-plate spline.

Two stages, deliberately separated.

**Stage 1 - similarity** (``estimateAffinePartial2D``): rotation, uniform scale
and translation, 4 degrees of freedom, RANSAC-robust. This is the single biggest
quality jump over the ratio-based placement in ``backend/overlay_test.py``,
because it derives the transform from measured correspondences instead of fixed
multipliers, so torso tilt and camera distance stop mattering.

Similarity rather than full affine is a load-bearing choice. At 90 and 270
degrees every torso landmark projects onto one vertical line - left and right
shoulder land on the same x. A full 6-DOF affine is degenerate on collinear
input and will produce garbage or fail outright. A 4-DOF similarity stays
determined: spacing along the line still fixes scale, and the line direction
still fixes rotation.

**Stage 2 - thin-plate spline**: non-rigid refinement so sleeves and hem follow
the body rather than staying rigid. Optional, because it is the expensive half -
the live path can skip it and the offline render pass never does.

OpenCV TPS argument order is a genuine trap, in two separate ways. Both verified
empirically; both pinned by ``backend/tests/test_warp.py``.

1. For images::

       tps.estimateTransformation(dst_points, src_points, matches)
       warped = tps.warpImage(src_image)       # moves src_points -> dst_points

   The reverse order does not raise - it silently returns an empty image.

2. ``applyTransformation`` maps *first argument -> second argument*, which is the
   opposite of what ``warpImage`` does visually with the same transformer. So
   mapping points forward needs a second transformer estimated with the roles
   swapped. That is why ``warp_garment`` builds two.

Cost note: TPS ``warpImage`` measures ~75 ms on a 640x480 frame - about 13 fps,
too slow to sit in the live loop. Hence ``use_tps`` defaults off for live
preview and on for the offline render pass, which is the whole reason the
capture-then-render design pays for itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .garment import GarmentAngle
from .pose import GARMENT_LANDMARKS, PoseResult

#: Below this visibility a landmark is dropped from the fit. Side views legitimately
#: occlude one shoulder, and a hallucinated landmark is worse than a missing one.
MIN_VISIBILITY = 0.35

#: TPS smoothing. 0 interpolates every correspondence exactly, which turns
#: landmark jitter into visible garment wobble. Small positive values trade a
#: little fidelity for a lot of temporal stability.
TPS_REGULARIZATION = 0.35


@dataclass
class Correspondence:
    """Matched garment-space and frame-space points."""

    names: List[str]
    src: np.ndarray  # (N, 2) garment image pixels
    dst: np.ndarray  # (N, 2) frame pixels

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class WarpResult:
    """A garment warped into frame space."""

    image: np.ndarray  # BGRA, frame-sized
    affine: np.ndarray  # 2x3
    residual_px: float
    n_points: int
    used_tps: bool


def build_correspondences(
    angle: GarmentAngle,
    pose: PoseResult,
    min_visibility: float = MIN_VISIBILITY,
    landmarks: Sequence[str] = GARMENT_LANDMARKS,
) -> Optional[Correspondence]:
    """Pair garment anchors with live body landmarks.

    Both sides key on MediaPipe landmark names, so this is a dict intersection.
    Anything the garment doesn't define, or the body can't see confidently, is
    dropped rather than guessed.
    """
    names: List[str] = []
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    for name in landmarks:
        garment_pt = angle.anchor(name)
        if garment_pt is None:
            continue
        body_pt = pose.point(name)
        if body_pt is None or pose.vis(name) < min_visibility:
            continue
        names.append(name)
        src.append(np.asarray(garment_pt, dtype=np.float64))
        dst.append(np.asarray(body_pt, dtype=np.float64))

    if len(names) < 2:  # estimateAffinePartial2D needs at least two pairs
        return None
    return Correspondence(names=names, src=np.array(src), dst=np.array(dst))


def fit_similarity(corr: Correspondence) -> Optional[np.ndarray]:
    """Robust 4-DOF similarity fit. Returns a 2x3 matrix or None."""
    src = corr.src.astype(np.float32)
    dst = corr.dst.astype(np.float32)

    matrix = None
    if len(corr) >= 3:
        matrix, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=12.0, maxIters=2000, refineIters=20
        )
    if matrix is None:
        # RANSAC needs elbow room it doesn't have at two or three points; a plain
        # least-squares fit is exactly determined there and never returns None
        # for non-coincident input.
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return matrix


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.hstack([points, np.ones((len(points), 1))])
    return homo @ matrix.T


def _boundary_points(frame_size: Tuple[int, int]) -> np.ndarray:
    """Frame corners and edge midpoints, used as identity correspondences.

    Without these the spline is only constrained inside the landmark hull and
    extrapolates wildly beyond it, smearing the garment across the frame. Pinning
    the border makes the deformation decay to identity away from the body.
    """
    w, h = frame_size
    return np.array(
        [
            [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
            [w / 2, 0], [w / 2, h - 1], [0, h / 2], [w - 1, h / 2],
        ],
        dtype=np.float64,
    )


def _estimate_tps(
    first: np.ndarray, second: np.ndarray, regularization: float
) -> cv2.ThinPlateSplineShapeTransformer:
    """Estimate a spline. ``applyTransformation`` will map first -> second;
    ``warpImage`` moves image content the other way. See module docstring."""
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.setRegularizationParameter(regularization)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(first))]
    tps.estimateTransformation(
        first.reshape(1, -1, 2).astype(np.float32),
        second.reshape(1, -1, 2).astype(np.float32),
        matches,
    )
    return tps


def warp_garment(
    angle: GarmentAngle,
    pose: PoseResult,
    frame_size: Optional[Tuple[int, int]] = None,
    use_tps: bool = True,
    min_visibility: float = MIN_VISIBILITY,
    regularization: float = TPS_REGULARIZATION,
) -> Optional[WarpResult]:
    """Warp one garment angle onto the body in ``pose``.

    ``frame_size`` defaults to the pose's own frame. Returns None when there are
    too few usable correspondences to fit anything.
    """
    corr = build_correspondences(angle, pose, min_visibility=min_visibility)
    if corr is None:
        return None

    size = frame_size or pose.frame_size
    matrix = fit_similarity(corr)
    if matrix is None:
        return None

    warped = cv2.warpAffine(
        angle.image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    landed = _apply_affine(matrix, corr.src)
    used_tps = False

    if use_tps and len(corr) >= 3:
        boundary = _boundary_points(size)
        tps_src = np.vstack([landed, boundary])
        tps_dst = np.vstack([corr.dst, boundary])
        try:
            # Image transformer: warpImage moves content src -> dst.
            image_tps = _estimate_tps(tps_dst, tps_src, regularization)
            warped = image_tps.warpImage(warped)
            # Point transformer: roles swapped so applyTransformation goes forward.
            point_tps = _estimate_tps(tps_src, tps_dst, regularization)
            landed = _tps_apply(point_tps, landed)
            used_tps = True
        except cv2.error:
            # Degenerate point configurations (all correspondences collinear and
            # coincident) can make the spline solve fail. The similarity result
            # is already usable, so keep it rather than dropping the frame.
            pass

    residual = float(np.mean(np.linalg.norm(landed - corr.dst, axis=1)))
    return WarpResult(
        image=warped,
        affine=matrix,
        residual_px=residual,
        n_points=len(corr),
        used_tps=used_tps,
    )


def _tps_apply(tps: cv2.ThinPlateSplineShapeTransformer, points: np.ndarray) -> np.ndarray:
    """Push points through an estimated TPS, matching warpImage's direction."""
    shaped = points.reshape(1, -1, 2).astype(np.float32)
    _, out = tps.applyTransformation(shaped)
    return np.asarray(out).reshape(-1, 2).astype(np.float64)
