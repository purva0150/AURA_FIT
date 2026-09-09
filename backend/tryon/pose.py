"""MediaPipe Pose wrapper.

Everything downstream (yaw estimation, warping, capture) talks to this module
rather than to MediaPipe directly, so landmark names, pixel conversion and
visibility gating are handled in exactly one place.

Two coordinate spaces come out of a single inference pass and both matter:

- ``px``    - landmarks projected into frame pixels. Used for warping, because
              that is the space the garment has to land in.
- ``world`` - metric 3D in metres, origin at the hip midpoint. Used for yaw,
              because it is the only output that actually carries depth.

MediaPipe's depth convention: smaller z is *closer* to the camera. `yaw.py`
depends on that sign, so don't normalise it away here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils

#: All 33 landmark names, lowercased (``left_shoulder``, ``nose``, ...).
LANDMARK_NAMES: Tuple[str, ...] = tuple(lm.name.lower() for lm in _mp_pose.PoseLandmark)

#: The four points that define torso orientation.
TORSO_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

#: Face points. Their collective visibility is what tells us whether the person
#: is facing the camera or away from it.
FACE_LANDMARKS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

#: The landmarks a garment anchor JSON is allowed to key on. ``neck`` is derived
#: (shoulder midpoint) rather than a real MediaPipe landmark - see PoseResult.neck.
GARMENT_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "neck",
    "left_chest",
    "right_chest",
    "left_waist",
    "right_waist",
)

DEFAULT_VISIBILITY = 0.5


@dataclass(frozen=True)
class PoseResult:
    """One frame's worth of pose, in both pixel and metric space."""

    px: Dict[str, np.ndarray]
    world: Dict[str, np.ndarray]
    visibility: Dict[str, float]
    segmentation: Optional[np.ndarray]
    frame_size: Tuple[int, int]  # (width, height)

    # -- pixel space -----------------------------------------------------

    def point(self, name: str) -> Optional[np.ndarray]:
        """Pixel coords of ``name``, or None. Understands the derived ``neck``."""
        if name == "neck":
            return self.neck()
        return self.px.get(name)

    def neck(self) -> Optional[np.ndarray]:
        """Shoulder midpoint. Not a MediaPipe landmark - derived so that garment
        collars have something stable to anchor to."""
        return self.midpoint("left_shoulder", "right_shoulder")

    def midpoint(self, a: str, b: str) -> Optional[np.ndarray]:
        pa, pb = self.px.get(a), self.px.get(b)
        if pa is None or pb is None:
            return None
        return (pa + pb) / 2.0

    # -- visibility ------------------------------------------------------

    def vis(self, name: str) -> float:
        """Visibility in [0, 1]. ``neck`` inherits the weaker of its two parents,
        so a half-occluded shoulder pair can't produce a confident collar."""
        if name == "neck":
            return min(self.vis("left_shoulder"), self.vis("right_shoulder"))
        return float(self.visibility.get(name, 0.0))

    def visible(self, name: str, thresh: float = DEFAULT_VISIBILITY) -> bool:
        return self.vis(name) >= thresh

    def mean_visibility(self, names: Iterable[str]) -> float:
        vals = [self.vis(n) for n in names]
        return float(np.mean(vals)) if vals else 0.0

    # -- metric space ----------------------------------------------------

    def world_point(self, name: str) -> Optional[np.ndarray]:
        if name == "neck":
            wl, wr = self.world.get("left_shoulder"), self.world.get("right_shoulder")
            if wl is None or wr is None:
                return None
            return (wl + wr) / 2.0
        return self.world.get(name)

    # -- convenience -----------------------------------------------------

    def torso_height_px(self) -> Optional[float]:
        """Shoulder-midpoint to hip-midpoint distance. This is the one scale cue
        that survives every rotation - horizontal spans collapse at 90 degrees,
        vertical span does not."""
        top = self.midpoint("left_shoulder", "right_shoulder")
        bottom = self.midpoint("left_hip", "right_hip")
        if top is None or bottom is None:
            return None
        return float(np.linalg.norm(bottom - top))

    def shoulder_width_px(self) -> Optional[float]:
        pl, pr = self.px.get("left_shoulder"), self.px.get("right_shoulder")
        if pl is None or pr is None:
            return None
        return float(np.linalg.norm(pl - pr))


class PoseEstimator:
    """Context-managed MediaPipe Pose.

    ``segmentation=True`` costs a few ms and gives a person mask we reuse for
    cheap occlusion in the live path (the offline path uses SegFormer instead).
    """

    def __init__(
        self,
        *,
        segmentation: bool = True,
        model_complexity: int = 1,
        static: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._pose = _mp_pose.Pose(
            static_image_mode=static,
            model_complexity=model_complexity,
            smooth_landmarks=not static,
            enable_segmentation=segmentation,
            smooth_segmentation=not static,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._pose.close()

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """Run inference on a BGR frame. Returns None when no person is found."""
        h, w = frame_bgr.shape[:2]
        results = self._pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        px: Dict[str, np.ndarray] = {}
        visibility: Dict[str, float] = {}
        for name, lm in zip(LANDMARK_NAMES, results.pose_landmarks.landmark):
            px[name] = np.array([lm.x * w, lm.y * h], dtype=np.float64)
            visibility[name] = float(lm.visibility)

        world: Dict[str, np.ndarray] = {}
        if results.pose_world_landmarks:
            for name, lm in zip(LANDMARK_NAMES, results.pose_world_landmarks.landmark):
                world[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)

        seg = None
        if results.segmentation_mask is not None:
            seg = np.asarray(results.segmentation_mask, dtype=np.float32)

        return PoseResult(
            px=px,
            world=world,
            visibility=visibility,
            segmentation=seg,
            frame_size=(w, h),
        )


def draw_skeleton(frame_bgr: np.ndarray, result: PoseResult, color=(0, 220, 120)) -> np.ndarray:
    """Minimal skeleton draw. We rebuild it from PoseResult rather than calling
    mp_drawing so the debug view reflects exactly the points warping will use."""
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ]
    for a, b in connections:
        pa, pb = result.point(a), result.point(b)
        if pa is None or pb is None:
            continue
        if result.vis(a) < 0.3 or result.vis(b) < 0.3:
            continue
        cv2.line(frame_bgr, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2)

    for name in GARMENT_LANDMARKS:
        p = result.point(name)
        if p is None:
            continue
        # Filled when confident, hollow when the warp will be dropping this pair.
        confident = result.visible(name)
        cv2.circle(frame_bgr, tuple(p.astype(int)), 5, color, -1 if confident else 1)
    return frame_bgr
