"""Garment warping: correspondences -> similarity fit -> thin-plate spline.

Two stages, deliberately separated.

Stage 1 - similarity:
    Rotation, uniform scale and translation using measured body/garment
    correspondences.

Stage 2 - thin-plate spline:
    Non-rigid refinement so sleeves and hem can follow the body.

The live pipeline can disable TPS for better FPS, while the offline renderer
can enable it for higher-quality deformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .garment import GarmentAngle
from .pose import GARMENT_LANDMARKS, PoseResult


#: Below this visibility a landmark is dropped from the fit.
MIN_VISIBILITY = 0.35


#: TPS smoothing.
#: Small positive values reduce visible temporal wobble caused by
#: landmark jitter.
TPS_REGULARIZATION = 0.35


# Stable torso landmarks used for the main similarity fit.
#
# Elbows are intentionally excluded from the rigid similarity stage because
# arm movement should not change the overall size/position of the garment.
#
# Elbows are still retained in the full correspondence set and are therefore
# available to TPS for sleeve deformation.
SIMILARITY_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


@dataclass
class Correspondence:
    """Matched garment-space and frame-space points."""

    names: List[str]

    # Garment image coordinates.
    src: np.ndarray  # shape: (N, 2)

    # Live camera/body coordinates.
    dst: np.ndarray  # shape: (N, 2)

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class WarpResult:
    """A garment warped into frame space."""

    # Frame-sized BGRA image.
    image: np.ndarray

    # 2x3 similarity transformation matrix.
    affine: np.ndarray

    # Average landmark error in pixels.
    residual_px: float

    # Number of usable body/garment correspondences.
    n_points: int

    # True when TPS refinement was successfully applied.
    used_tps: bool


def build_correspondences(
    angle: GarmentAngle,
    pose: PoseResult,
    min_visibility: float = MIN_VISIBILITY,
    landmarks: Sequence[str] = GARMENT_LANDMARKS,
    fit_scale: float = 1.0,
) -> Optional[Correspondence]:
    """Pair garment anchors with live body landmarks.

    Both sides use MediaPipe landmark names.

    A landmark is used only when:
        1. the garment has that anchor,
        2. the body landmark exists,
        3. the landmark visibility is high enough.

    Missing or unreliable landmarks are simply ignored.

    At least two correspondences are required because OpenCV's similarity
    estimation needs enough information to determine a transform.
    """

    names: List[str] = []
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    # Joint centres are not garment boundaries. Preserve the catalogue
    # garment's proportions so a seated pelvis does not pinch the hem and a
    # short sleeve is guided by (rather than stretched to) the elbow.
    body_targets: dict[str, np.ndarray] = {}
    src_ls, src_rs = angle.anchor("left_shoulder"), angle.anchor("right_shoulder")
    src_lh, src_rh = angle.anchor("left_hip"), angle.anchor("right_hip")
    dst_ls, dst_rs = pose.point("left_shoulder"), pose.point("right_shoulder")
    dst_lh, dst_rh = pose.point("left_hip"), pose.point("right_hip")

    if all(p is not None for p in (src_ls, src_rs, src_lh, src_rh, dst_ls, dst_rs, dst_lh, dst_rh)):
        source_shoulder_width = float(np.linalg.norm(src_ls - src_rs))
        source_hem_width = float(np.linalg.norm(src_lh - src_rh))
        body_axis = np.asarray(dst_ls - dst_rs, dtype=np.float64)
        body_shoulder_width = float(np.linalg.norm(body_axis))
        if source_shoulder_width > 1.0 and body_shoulder_width > 1.0:
            axis = body_axis / body_shoulder_width
            hip_center = (np.asarray(dst_lh) + np.asarray(dst_rh)) * 0.5
            shoulder_center = (np.asarray(dst_ls) + np.asarray(dst_rs)) * 0.5
            torso_vector = hip_center - shoulder_center
            torso_length = float(np.linalg.norm(torso_vector))
            down = torso_vector / max(torso_length, 1.0)
            hem_extension = float(np.clip(angle.fit.get("hem_extension", 0.12), 0.0, 0.85))
            fit_scale = float(np.clip(fit_scale, 0.90, 1.10))
            hem_center = shoulder_center + torso_vector * ((1.0 + hem_extension) * fit_scale)
            source_width_ratio = source_hem_width / source_shoulder_width
            configured_width = float(angle.fit.get("hem_width_ratio", source_width_ratio))
            target_width = body_shoulder_width * float(np.clip(configured_width * fit_scale, 0.72, 1.18))
            body_targets["left_hip"] = hem_center + axis * target_width * 0.5
            body_targets["right_hip"] = hem_center - axis * target_width * 0.5

            # Extra torso controls stop TPS from treating the whole chest and
            # waist as one rubber sheet between shoulders and hem.
            chest_center = shoulder_center + torso_vector * 0.38
            waist_center = shoulder_center + torso_vector * 0.72
            observed_hip_width = float(np.linalg.norm(np.asarray(dst_lh)-np.asarray(dst_rh)))
            chest_width = body_shoulder_width * float(np.clip(angle.fit.get("chest_width_ratio", 0.94) * fit_scale, 0.72, 1.10))
            configured_waist = body_shoulder_width * float(np.clip(angle.fit.get("waist_width_ratio", 0.88) * fit_scale, 0.68, 1.08))
            # Blend the garment cut with the live hip span. Hip landmarks are
            # joint centres rather than the silhouette, so they guide instead
            # of fully determining the width.
            waist_width = configured_waist*.72 + observed_hip_width*.28
            body_targets["left_chest"] = chest_center + axis * chest_width * 0.5
            body_targets["right_chest"] = chest_center - axis * chest_width * 0.5
            body_targets["left_waist"] = waist_center + axis * waist_width * 0.5
            body_targets["right_waist"] = waist_center - axis * waist_width * 0.5

        source_torso_height = float(
            np.linalg.norm((src_lh + src_rh) * 0.5 - (src_ls + src_rs) * 0.5)
        )
        body_torso_height = float(
            np.linalg.norm((dst_lh + dst_rh) * 0.5 - (dst_ls + dst_rs) * 0.5)
        )
        if source_torso_height > 1.0 and body_torso_height > 1.0:
            for side, source_shoulder, body_shoulder in (
                ("left", src_ls, dst_ls), ("right", src_rs, dst_rs)
            ):
                source_elbow = angle.anchor(f"{side}_elbow")
                body_elbow = pose.point(f"{side}_elbow")
                if source_elbow is None or body_elbow is None:
                    continue
                direction = np.asarray(body_elbow - body_shoulder, dtype=np.float64)
                actual_length = float(np.linalg.norm(direction))
                if actual_length < 1.0:
                    continue
                sleeve_ratio = float(angle.fit.get(
                    "sleeve_length_ratio",
                    float(np.linalg.norm(source_elbow - source_shoulder)) / source_torso_height,
                ))
                sleeve_length = min(actual_length * 0.78, body_torso_height * sleeve_ratio * fit_scale)
                body_targets[f"{side}_elbow"] = (
                    np.asarray(body_shoulder) + direction / actual_length * sleeve_length
                )

    for name in landmarks:
        # Garment anchor.
        garment_pt = angle.anchor(name)

        if garment_pt is None:
            continue

        # Body landmark.
        body_pt = body_targets.get(name, pose.point(name))

        if body_pt is None:
            continue

        # MediaPipe visibility check.
        if name in body_targets:
            target_visibility = min(
                pose.vis("left_shoulder"), pose.vis("right_shoulder"),
                pose.vis("left_hip"), pose.vis("right_hip"),
            )
        else:
            target_visibility = pose.vis(name)
        if target_visibility < min_visibility:
            continue

        names.append(name)

        src.append(
            np.asarray(
                garment_pt,
                dtype=np.float64,
            )
        )

        dst.append(
            np.asarray(
                body_pt,
                dtype=np.float64,
            )
        )

    # Similarity estimation needs at least two usable point pairs.
    if len(names) < 2:
        return None

    return Correspondence(
        names=names,
        src=np.array(src, dtype=np.float64),
        dst=np.array(dst, dtype=np.float64),
    )


def fit_similarity(
    corr: Correspondence,
) -> Optional[np.ndarray]:
    """Estimate a robust 4-DOF similarity transformation.

    The transformation contains:

        - rotation
        - uniform scale
        - translation

    Full affine deformation is intentionally avoided because side views can
    produce nearly collinear landmark configurations.
    """

    src = corr.src.astype(np.float32)
    dst = corr.dst.astype(np.float32)

    matrix = None

    # With 3+ points, use RANSAC so one bad landmark does not destroy
    # the entire garment placement.
    if len(corr) >= 3:
        matrix, _ = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=12.0,
            maxIters=2000,
            refineIters=20,
        )

    # Fallback for small/degenerate correspondence sets.
    if matrix is None:
        matrix, _ = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.LMEDS,
        )

    return matrix


def _apply_affine(
    matrix: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points."""

    homo = np.hstack(
        [
            points,
            np.ones(
                (len(points), 1),
                dtype=np.float64,
            ),
        ]
    )

    return homo @ matrix.T


def _boundary_points(
    frame_size: Tuple[int, int],
) -> np.ndarray:
    """Create frame boundary control points for TPS.

    Pinning the frame boundary to itself prevents the TPS from extrapolating
    wildly outside the body/garment region.
    """

    w, h = frame_size

    return np.array(
        [
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1],

            [w / 2, 0],
            [w / 2, h - 1],
            [0, h / 2],
            [w - 1, h / 2],
        ],
        dtype=np.float64,
    )


def _estimate_tps(
    first: np.ndarray,
    second: np.ndarray,
    regularization: float,
) -> cv2.ThinPlateSplineShapeTransformer:
    """Estimate a Thin Plate Spline transformer.

    OpenCV's TPS API has unintuitive argument directions, so this helper keeps
    the ordering in one place.

    ``applyTransformation`` maps first -> second.

    ``warpImage`` moves image content in the corresponding visual direction
    using the estimated transformer.
    """

    factory = getattr(cv2, "createThinPlateSplineShapeTransformer", None)
    if factory is None:
        raise RuntimeError(
            "Thin-plate-spline support is unavailable. Install "
            "opencv-contrib-python (not opencv-python)."
        )

    tps = factory()

    tps.setRegularizationParameter(
        regularization
    )

    matches = [
        cv2.DMatch(i, i, 0)
        for i in range(len(first))
    ]

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
    fit_scale: float = 1.0,
) -> Optional[WarpResult]:
    """Warp one garment angle onto the detected body.

    Pipeline:

        garment anchors
              ↓
        body landmarks
              ↓
        torso similarity transform
              ↓
        affine garment
              ↓
        optional TPS deformation
              ↓
        frame-sized warped garment

    The important design choice here is that torso landmarks are used for
    the rigid similarity fit while the complete landmark set is retained
    for TPS.

    This prevents moving elbows from changing the garment's overall size,
    while still allowing TPS to bend the sleeves toward the arms.
    """

    # ------------------------------------------------------------
    # STEP 1
    # Build torso-only correspondences for the similarity transform.
    #
    # These landmarks describe the torso and are comparatively stable:
    #
    #   left shoulder
    #   right shoulder
    #   left hip
    #   right hip
    #   neck
    #
    # Elbows are intentionally excluded here.
    # ------------------------------------------------------------

    fit_corr = build_correspondences(
        angle,
        pose,
        min_visibility=min_visibility,
        landmarks=SIMILARITY_LANDMARKS,
        fit_scale=fit_scale,
    )

    # ------------------------------------------------------------
    # STEP 2
    # Build the complete correspondence set.
    #
    # This includes elbows and is used later by TPS.
    # ------------------------------------------------------------

    corr = build_correspondences(
        angle,
        pose,
        min_visibility=min_visibility,
        landmarks=GARMENT_LANDMARKS,
        fit_scale=fit_scale,
    )

    # If either set is unavailable, there is not enough information
    # to safely place the garment.
    if fit_corr is None or corr is None:
        return None

    # ------------------------------------------------------------
    # STEP 3
    # Determine output frame size.
    # ------------------------------------------------------------

    size = frame_size or pose.frame_size

    # ------------------------------------------------------------
    # STEP 4
    # Estimate the rigid similarity transform using ONLY torso points.
    # ------------------------------------------------------------

    matrix = fit_similarity(
        fit_corr
    )

    if matrix is None:
        return None

    # ------------------------------------------------------------
    # STEP 5
    # Apply the similarity transform to the entire garment image.
    # ------------------------------------------------------------

    warped = cv2.warpAffine(
        angle.image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Apply the same similarity transform to all correspondence points.
    #
    # This gives us the current location of each garment anchor after
    # the rigid transformation.
    landed = _apply_affine(
        matrix,
        corr.src,
    )

    used_tps = False

    # ------------------------------------------------------------
    # STEP 6
    # Optional Thin Plate Spline refinement.
    # ------------------------------------------------------------

    if use_tps and len(corr) >= 3:

        # Keep the image boundary stable so TPS does not deform the
        # entire frame.
        boundary = _boundary_points(size)

        # Current transformed garment points.
        #
        # Example:
        #
        # garment shoulder
        #       ↓
        # similarity transform
        #       ↓
        # landed shoulder
        #
        # We now want:
        #
        # landed shoulder
        #       ↓
        # TPS
        #       ↓
        # actual body shoulder
        #
        tps_src = np.vstack(
            [
                landed,
                boundary,
            ]
        )

        tps_dst = np.vstack(
            [
                corr.dst,
                boundary,
            ]
        )

        try:
            # ----------------------------------------------------
            # IMAGE TRANSFORMER
            #
            # For warpImage, OpenCV expects the arguments in the
            # reverse visual direction.
            # ----------------------------------------------------

            image_tps = _estimate_tps(
                tps_dst,
                tps_src,
                regularization,
            )

            warped = image_tps.warpImage(
                warped
            )

            # ----------------------------------------------------
            # POINT TRANSFORMER
            #
            # We need to calculate where the transformed landmark
            # points actually ended up.
            #
            # Therefore create a second TPS with the roles swapped.
            # ----------------------------------------------------

            point_tps = _estimate_tps(
                tps_src,
                tps_dst,
                regularization,
            )

            landed = _tps_apply(
                point_tps,
                landed,
            )

            used_tps = True

        except (cv2.error, RuntimeError, AttributeError):
            # TPS can fail for degenerate configurations, especially
            # when points become nearly collinear.
            #
            # The similarity result is still valid, so keep it.
            used_tps = False

    # ------------------------------------------------------------
    # STEP 7
    # Calculate final landmark residual.
    #
    # Smaller residual = garment anchors are closer to their intended
    # body landmarks.
    # ------------------------------------------------------------

    residual = float(
        np.mean(
            np.linalg.norm(
                landed - corr.dst,
                axis=1,
            )
        )
    )

    # ------------------------------------------------------------
    # STEP 8
    # Return complete warp result.
    # ------------------------------------------------------------

    return WarpResult(
        image=warped,
        affine=matrix,
        residual_px=residual,
        n_points=len(corr),
        used_tps=used_tps,
    )


def _tps_apply(
    tps: cv2.ThinPlateSplineShapeTransformer,
    points: np.ndarray,
) -> np.ndarray:
    """Push points through an estimated TPS.

    Returns an Nx2 float64 array.
    """

    shaped = points.reshape(
        1,
        -1,
        2,
    ).astype(np.float32)

    _, out = tps.applyTransformation(
        shaped
    )

    return (
        np.asarray(out)
        .reshape(-1, 2)
        .astype(np.float64)
    )
