"""Body yaw estimation.

This replaces the shoulder-width proxy in ``backend/pose_test.py``. That proxy
cannot distinguish "turned 45 degrees" from "standing twice as far away", which
makes it useless for picking a garment angle. Metric world landmarks carry real
depth, so the angle can be measured instead of guessed.

Geometry
--------
Take the vector from the right shoulder to the left shoulder in world space::

    v = world[left_shoulder] - world[right_shoulder]

MediaPipe world axes: +x is image-right, +y is down, and z is depth where
**smaller means closer to the camera**.

- Facing the camera: the person's left shoulder is on image-right, so
  ``v.x > 0`` and ``v.z ~ 0``  ->  ``atan2(v.z, v.x) = 0``.
- Turned 90 degrees with their right side to the camera: the right shoulder is
  nearer, so ``right.z < left.z``, giving ``v.z > 0`` while ``v.x ~ 0``
  ->  ``atan2(v.z, v.x) = +90``.
- Back to the camera: left shoulder is on image-left, ``v.x < 0``, ``v.z ~ 0``
  ->  ``180``.

So ``yaw = atan2(v.z, v.x)`` falls out directly, and the convention is:

    **0 = facing camera, increasing = turning to their own left, 180 = back.**

The hip vector gives an independent read of the same angle. We average the two
(weighted by landmark visibility) and treat their disagreement as a confidence
penalty, since a large split means either torso twist or noisy depth.

Near 90 and 270 degrees ``v.x`` collapses and the estimate rides entirely on
``v.z``, which is the noisiest channel MediaPipe produces. That is exactly where
the smoother and the confidence gate earn their keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .pose import FACE_LANDMARKS, PoseResult

#: The eight capture angles. Degrees, using the convention documented above.
CAPTURE_BINS: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

#: Expected shoulder separation in metres. Used only to sanity-check that the
#: world landmarks are dimensionally plausible before trusting their depth.
_TYPICAL_SHOULDER_SPAN_M = 0.35


def normalize_deg(deg: float) -> float:
    """Wrap to [0, 360)."""
    return deg % 360.0


def angular_diff(a: float, b: float) -> float:
    """Signed shortest angular distance ``a - b``, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def angular_dist(a: float, b: float) -> float:
    """Unsigned shortest angular distance, in [0, 180]."""
    return abs(angular_diff(a, b))


@dataclass(frozen=True)
class YawEstimate:
    """A single frame's yaw reading."""

    degrees: float
    confidence: float
    facing_camera: bool
    shoulder_degrees: Optional[float] = None
    hip_degrees: Optional[float] = None
    flipped: bool = False  # True when the face check overrode the raw angle

    @property
    def radians(self) -> float:
        return math.radians(self.degrees)


def _axis_yaw(pose: PoseResult, left: str, right: str) -> Optional[Tuple[float, float]]:
    """Yaw (degrees) and horizontal span (metres) from one left/right pair."""
    wl, wr = pose.world_point(left), pose.world_point(right)
    if wl is None or wr is None:
        return None
    v = wl - wr
    dx, dz = float(v[0]), float(v[2])
    span = math.hypot(dx, dz)
    if span < 1e-4:
        return None
    return normalize_deg(math.degrees(math.atan2(dz, dx))), span


def _facing_camera(pose: PoseResult) -> Tuple[bool, float]:
    """Is the person facing us? Returns (facing, strength in [0, 1]).

    Two independent signals, because either alone is flaky:

    - Face landmark visibility. Collapses when someone turns away.
    - Nose depth versus shoulder-midpoint depth. Purely geometric, so it holds
      up when visibility is confused by hair or hats.
    """
    vis = pose.mean_visibility(FACE_LANDMARKS)

    depth_vote = 0.0
    nose = pose.world_point("nose")
    neck = pose.world_point("neck")
    if nose is not None and neck is not None:
        # Smaller z is closer, so a nose in front of the shoulders reads negative.
        delta = float(neck[2] - nose[2])
        depth_vote = float(np.clip(delta / 0.12, -1.0, 1.0))

    vis_vote = float(np.clip((vis - 0.5) * 2.0, -1.0, 1.0))
    score = 0.5 * vis_vote + 0.5 * depth_vote
    return score >= 0.0, abs(score)


def estimate_yaw(pose: PoseResult) -> Optional[YawEstimate]:
    """Estimate torso yaw in degrees, or None if the landmarks can't support it."""
    shoulder = _axis_yaw(pose, "left_shoulder", "right_shoulder")
    hip = _axis_yaw(pose, "left_hip", "right_hip")
    if shoulder is None and hip is None:
        return None

    readings: list[Tuple[float, float]] = []  # (degrees, weight)
    if shoulder is not None:
        w = pose.mean_visibility(("left_shoulder", "right_shoulder"))
        readings.append((shoulder[0], max(w, 1e-3)))
    if hip is not None:
        w = pose.mean_visibility(("left_hip", "right_hip"))
        # Hips sit closer to the world-landmark origin and are noisier in depth,
        # so they get a standing discount rather than an equal vote.
        readings.append((hip[0], max(w, 1e-3) * 0.7))

    # Average on the unit circle - a plain mean would break across the 0/360 seam.
    vec = np.zeros(2)
    for deg, weight in readings:
        rad = math.radians(deg)
        vec += weight * np.array([math.cos(rad), math.sin(rad)])
    if np.linalg.norm(vec) < 1e-6:
        return None
    raw_deg = normalize_deg(math.degrees(math.atan2(vec[1], vec[0])))

    facing, facing_strength = _facing_camera(pose)

    # The raw angle already spans a full 360, but its front/back half depends on
    # the sign of v.x - which flips if MediaPipe mislabels left and right on a
    # back view. When the face evidence is strong and contradicts the raw angle,
    # trust the face and rotate by 180.
    flipped = False
    reads_as_front = angular_dist(raw_deg, 0.0) < 60.0
    reads_as_back = angular_dist(raw_deg, 180.0) < 60.0
    if facing_strength > 0.4:
        if reads_as_front and not facing:
            raw_deg, flipped = normalize_deg(raw_deg + 180.0), True
        elif reads_as_back and facing:
            raw_deg, flipped = normalize_deg(raw_deg + 180.0), True

    confidence = _confidence(pose, shoulder, hip, readings)

    return YawEstimate(
        degrees=raw_deg,
        confidence=confidence,
        facing_camera=facing,
        shoulder_degrees=shoulder[0] if shoulder else None,
        hip_degrees=hip[0] if hip else None,
        flipped=flipped,
    )


def _confidence(
    pose: PoseResult,
    shoulder: Optional[Tuple[float, float]],
    hip: Optional[Tuple[float, float]],
    readings: Sequence[Tuple[float, float]],
) -> float:
    """Blend three penalties into a single [0, 1] score."""
    vis = pose.mean_visibility(("left_shoulder", "right_shoulder", "left_hip", "right_hip"))

    # Dimensional plausibility: if the metric shoulder span is nowhere near a
    # real shoulder width, the world landmarks aren't converged and their depth
    # channel is not worth reading.
    span_score = 1.0
    if shoulder is not None:
        ratio = shoulder[1] / _TYPICAL_SHOULDER_SPAN_M
        span_score = float(np.clip(1.0 - abs(math.log(max(ratio, 1e-3))), 0.0, 1.0))

    # Agreement between the shoulder and hip reads.
    agree_score = 1.0
    if shoulder is not None and hip is not None:
        agree_score = float(np.clip(1.0 - angular_dist(shoulder[0], hip[0]) / 60.0, 0.0, 1.0))

    return float(np.clip(0.5 * vis + 0.3 * agree_score + 0.2 * span_score, 0.0, 1.0))


class YawSmoother:
    """Exponential moving average on the unit circle.

    Averaging degrees directly would tear across the 0/360 seam, so the state is
    a 2D vector. Low-confidence frames contribute proportionally less instead of
    being dropped outright, which keeps the reading alive through the side views
    where confidence legitimately dips.
    """

    def __init__(self, alpha: float = 0.25, min_confidence: float = 0.15) -> None:
        self.alpha = alpha
        self.min_confidence = min_confidence
        self._vec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._vec = None

    def update(self, est: Optional[YawEstimate]) -> Optional[float]:
        if est is None or est.confidence < self.min_confidence:
            return self.value
        rad = math.radians(est.degrees)
        sample = np.array([math.cos(rad), math.sin(rad)])
        a = self.alpha * est.confidence
        self._vec = sample if self._vec is None else (1 - a) * self._vec + a * sample
        return self.value

    @property
    def value(self) -> Optional[float]:
        if self._vec is None or np.linalg.norm(self._vec) < 1e-9:
            return None
        return normalize_deg(math.degrees(math.atan2(self._vec[1], self._vec[0])))

    @property
    def stability(self) -> float:
        """Length of the averaged vector, in [0, 1]. Near 1 means the recent
        readings agree; a dip means the person is mid-turn or the estimate is
        thrashing. The capture state machine uses this to decide when a pose has
        settled enough to keep."""
        if self._vec is None:
            return 0.0
        return float(np.clip(np.linalg.norm(self._vec), 0.0, 1.0))


@dataclass(frozen=True)
class BinBlend:
    """The two bins bracketing a yaw, with the weight owed to the upper one.

    Rendering both and cross-dissolving by ``weight`` is what stops the garment
    jump-cutting as the user rotates past a bin boundary.
    """

    lo: float
    hi: float
    weight: float

    @property
    def nearest(self) -> float:
        return self.hi if self.weight >= 0.5 else self.lo


def nearest_bin(yaw: float, bins: Sequence[float] = CAPTURE_BINS) -> float:
    return min(bins, key=lambda b: angular_dist(yaw, b))


def blend_bins(yaw: float, bins: Sequence[float] = CAPTURE_BINS) -> BinBlend:
    """Find the two bins bracketing ``yaw`` and the interpolation weight."""
    yaw = normalize_deg(yaw)
    ordered = sorted(normalize_deg(b) for b in bins)
    if len(ordered) == 1:
        return BinBlend(ordered[0], ordered[0], 0.0)

    lo = ordered[-1]
    hi = ordered[0]
    for i, b in enumerate(ordered):
        if b <= yaw:
            lo = b
            hi = ordered[(i + 1) % len(ordered)]

    span = (hi - lo) % 360.0
    if span < 1e-9:
        return BinBlend(lo, hi, 0.0)
    weight = ((yaw - lo) % 360.0) / span
    return BinBlend(lo, hi, float(np.clip(weight, 0.0, 1.0)))
