"""A virtual person, for testing the pipeline without a webcam.

Builds a ``PoseResult`` at any yaw using the same projection model as
``tools/make_synthetic.py``. Because the garment assets and this body are
generated from identical maths, a correct warp has a known-zero residual - so
warp error becomes a measured number rather than an eyeball judgement.

Used by the unit tests and by ``tools/render_smoke.py``.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tryon.pose import LANDMARK_NAMES, PoseResult  # noqa: E402

# Metric half-spans, roughly adult proportions (metres).
_M_SHOULDER = 0.19
_M_HIP = 0.11
_M_SHOULDER_Y = -0.30
_M_NOSE_Y = -0.55
_M_NOSE_FWD = 0.10
_M_ELBOW_REACH = 0.30
_M_ELBOW_Y = -0.05


def _project(u: float, w: float, theta: float) -> Tuple[float, float]:
    """Body frame (u = person's left, w = forward) -> (image_x, depth_z)."""
    return (
        u * math.cos(theta) + w * math.sin(theta),
        u * math.sin(theta) - w * math.cos(theta),
    )


def make_pose(
    yaw_deg: float,
    frame_size: Tuple[int, int] = (1280, 720),
    shoulder_px: float = 105.0,
    shoulder_y_px: float = 205.0,
    hip_y_px: float = 470.0,
    elbow_drop_px: float = 135.0,
    visibility: float = 0.95,
) -> PoseResult:
    """A ``PoseResult`` for a person standing at ``yaw_deg``."""
    theta = math.radians(yaw_deg)
    w_px, h_px = frame_size
    cx = w_px / 2.0

    px: Dict[str, np.ndarray] = {}
    world: Dict[str, np.ndarray] = {}
    vis: Dict[str, float] = {name: 0.0 for name in LANDMARK_NAMES}

    def place(name: str, u_m: float, y_m: float, fwd_m: float, u_px: float, y_px: float) -> None:
        ix, iz = _project(u_m, fwd_m, theta)
        world[name] = np.array([ix, y_m, iz], dtype=np.float64)
        sx, _ = _project(u_px, 0.0, theta)
        px[name] = np.array([cx + sx, y_px], dtype=np.float64)
        vis[name] = visibility

    hip_px = shoulder_px * (_M_HIP / _M_SHOULDER)
    elbow_px = shoulder_px * (_M_ELBOW_REACH / _M_SHOULDER)

    place("left_shoulder", _M_SHOULDER, _M_SHOULDER_Y, 0.0, shoulder_px, shoulder_y_px)
    place("right_shoulder", -_M_SHOULDER, _M_SHOULDER_Y, 0.0, -shoulder_px, shoulder_y_px)
    place("left_hip", _M_HIP, 0.0, 0.0, hip_px, hip_y_px)
    place("right_hip", -_M_HIP, 0.0, 0.0, -hip_px, hip_y_px)
    place("left_elbow", _M_ELBOW_REACH, _M_ELBOW_Y, 0.0, elbow_px, shoulder_y_px + elbow_drop_px)
    place("right_elbow", -_M_ELBOW_REACH, _M_ELBOW_Y, 0.0, -elbow_px, shoulder_y_px + elbow_drop_px)
    place("left_wrist", _M_ELBOW_REACH * 1.05, 0.12, 0.0, elbow_px * 1.05, shoulder_y_px + elbow_drop_px * 1.9)
    place("right_wrist", -_M_ELBOW_REACH * 1.05, 0.12, 0.0, -elbow_px * 1.05, shoulder_y_px + elbow_drop_px * 1.9)
    place("nose", 0.0, _M_NOSE_Y, _M_NOSE_FWD, 0.0, shoulder_y_px - 95.0)

    # Face landmarks share the nose's depth so the facing check has something to
    # read; their visibility falls off as the person turns away, mimicking the
    # real detector.
    face_vis = visibility * max(0.05, math.cos(theta) * 0.5 + 0.5)
    for name in ("left_eye", "right_eye", "left_ear", "right_ear"):
        side = 1.0 if name.startswith("left") else -1.0
        place(name, side * 0.06, _M_NOSE_Y - 0.02, _M_NOSE_FWD * 0.5, side * 30.0, shoulder_y_px - 105.0)
        vis[name] = face_vis
    vis["nose"] = face_vis

    return PoseResult(
        px=px,
        world=world,
        visibility=vis,
        segmentation=None,
        frame_size=frame_size,
    )
