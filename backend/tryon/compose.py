"""Compositing: alpha blending, angle cross-dissolve, and occlusion.

Three jobs.

**Alpha compositing** - vectorised over the whole frame. The per-pixel loop
equivalent in ``backend/overlay_test.py`` only worked because the garment was a
small axis-aligned rectangle; a warped garment can land anywhere, so this
operates on full frame-sized buffers.

**Cross-dissolve** - at a yaw between two available garment angles, both are
warped and blended by weight. This is what removes the jump cut as the user
turns. It matters most for catalogue sources like DeepFashion In-shop that only
supply three real angles, where a single-nearest pick would snap through 60
degree steps.

**Occlusion** - a forearm crossing the torso must appear *in front* of the
garment. Two tiers, matching the two quality tiers elsewhere:

- live: MediaPipe's person segmentation plus limb polygons built from landmarks.
  Cheap and approximate.
- offline: SegFormer's per-limb parse (``Left-arm``/``Right-arm``/``Face``), which
  is far more accurate but far too slow for 30 fps.

Both produce the same thing - a float mask where 1 means "this pixel belongs to
something in front of the garment" - so the compositor doesn't care which tier
produced it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pose import PoseResult

#: Landmark chains whose limbs can pass in front of the torso.
_ARM_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)


def alpha_composite(background_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Blend a frame-sized BGRA overlay onto a BGR frame."""
    if overlay_bgra.shape[:2] != background_bgr.shape[:2]:
        raise ValueError(
            f"size mismatch: overlay {overlay_bgra.shape[:2]} vs frame {background_bgr.shape[:2]}"
        )
    alpha = (overlay_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    blended = alpha * overlay_bgra[:, :, :3].astype(np.float32) + (1.0 - alpha) * background_bgr.astype(
        np.float32
    )
    return blended.astype(np.uint8)


def blend_overlays(a: Optional[np.ndarray], b: Optional[np.ndarray], weight: float) -> Optional[np.ndarray]:
    """Cross-dissolve two BGRA overlays. ``weight`` is b's share, in [0, 1].

    Colour is blended in premultiplied space. Straight-alpha blending would let
    the transparent side's arbitrary RGB bleed into the result and produce dark
    fringes wherever the two garments' silhouettes disagree - which is precisely
    at the boundary, where the dissolve is most visible.
    """
    if a is None:
        return b
    if b is None:
        return a
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 1e-3:
        return a
    if weight >= 1.0 - 1e-3:
        return b

    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    alpha_a, alpha_b = af[:, :, 3:4] / 255.0, bf[:, :, 3:4] / 255.0

    premul = (1.0 - weight) * af[:, :, :3] * alpha_a + weight * bf[:, :, :3] * alpha_b
    alpha_out = (1.0 - weight) * alpha_a + weight * alpha_b

    rgb = np.divide(premul, np.maximum(alpha_out, 1e-6))
    out = np.zeros_like(af)
    out[:, :, :3] = np.clip(rgb, 0, 255)
    out[:, :, 3:4] = np.clip(alpha_out * 255.0, 0, 255)
    return out.astype(np.uint8)


def limb_occlusion_mask(
    pose: PoseResult,
    frame_size: Tuple[int, int],
    min_visibility: float = 0.4,
    thickness_scale: float = 0.22,
) -> np.ndarray:
    """Approximate arm mask from landmarks alone, as a float mask in [0, 1].

    Arms are drawn as tapered capsules along the shoulder-elbow-wrist chain.
    Width is derived from shoulder span so it tracks the subject's distance from
    the camera instead of being a fixed pixel count.
    """
    w, h = frame_size
    mask = np.zeros((h, w), dtype=np.uint8)

    span = pose.shoulder_width_px() or (w * 0.25)
    radius = max(int(span * thickness_scale), 6)

    for chain in _ARM_CHAINS:
        points = []
        for name in chain:
            p = pose.point(name)
            if p is None or pose.vis(name) < min_visibility:
                break
            points.append(p.astype(int))
        if len(points) < 2:
            continue
        for i in range(len(points) - 1):
            # Forearms are thinner than upper arms.
            r = int(radius * (1.0 - 0.25 * i))
            cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), 255, thickness=2 * r)
            cv2.circle(mask, tuple(points[i]), r, 255, -1)
        cv2.circle(mask, tuple(points[-1]), int(radius * 0.7), 255, -1)

    return cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.25).astype(np.float32) / 255.0


def arms_in_front(pose: PoseResult, margin_m: float = 0.02) -> Tuple[bool, bool]:
    """Is each arm nearer the camera than the torso? Returns (left, right).

    An arm hanging at the side is beside the garment, not over it - masking it
    unconditionally would punch permanent holes in the sleeves. Metric depth
    answers this directly: smaller z is nearer.
    """
    neck = pose.world_point("neck")
    if neck is None:
        return (False, False)
    torso_z = float(neck[2])

    out = []
    for side in ("left", "right"):
        zs = [
            float(pose.world_point(f"{side}_{joint}")[2])
            for joint in ("elbow", "wrist")
            if pose.world_point(f"{side}_{joint}") is not None
        ]
        out.append(bool(zs) and min(zs) < torso_z - margin_m)
    return (out[0], out[1])


def person_occlusion_mask(
    pose: PoseResult,
    frame_size: Tuple[int, int],
    use_depth_gate: bool = True,
) -> np.ndarray:
    """Live-tier occlusion mask: limb capsules, gated by metric depth."""
    w, h = frame_size
    if not use_depth_gate:
        return limb_occlusion_mask(pose, frame_size)

    left_front, right_front = arms_in_front(pose)
    if not (left_front or right_front):
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    span = pose.shoulder_width_px() or (w * 0.25)
    radius = max(int(span * 0.22), 6)

    for chain, active in zip(_ARM_CHAINS, (left_front, right_front)):
        if not active:
            continue
        points = []
        for name in chain:
            p = pose.point(name)
            if p is None or pose.vis(name) < 0.4:
                break
            points.append(p.astype(int))
        if len(points) < 2:
            continue
        limb = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(points) - 1):
            r = int(radius * (1.0 - 0.25 * i))
            cv2.line(limb, tuple(points[i]), tuple(points[i + 1]), 255, thickness=2 * r)
            cv2.circle(limb, tuple(points[i]), r, 255, -1)
        cv2.circle(limb, tuple(points[-1]), int(radius * 0.7), 255, -1)
        blurred = cv2.GaussianBlur(limb, (0, 0), sigmaX=max(radius * 0.25, 1.0))
        mask = np.maximum(mask, blurred.astype(np.float32) / 255.0)

    # Constrain to actual person pixels so a mistracked wrist can't erase garment
    # over the background.
    if pose.segmentation is not None:
        seg = pose.segmentation
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        mask *= np.clip(seg, 0.0, 1.0)
    return mask


def apply_occlusion(overlay_bgra: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Knock ``mask`` out of the overlay's alpha channel."""
    if mask is None:
        return overlay_bgra
    if mask.shape[:2] != overlay_bgra.shape[:2]:
        mask = cv2.resize(mask, (overlay_bgra.shape[1], overlay_bgra.shape[0]))
    out = overlay_bgra.copy()
    keep = np.clip(1.0 - mask, 0.0, 1.0)
    out[:, :, 3] = (out[:, :, 3].astype(np.float32) * keep).astype(np.uint8)
    return out


def feather_alpha(overlay_bgra: np.ndarray, radius: int = 3) -> np.ndarray:
    """Soften the overlay's alpha edge so the composite doesn't show a hard cut."""
    if radius <= 0:
        return overlay_bgra
    out = overlay_bgra.copy()
    blurred = cv2.GaussianBlur(out[:, :, 3], (0, 0), sigmaX=radius)
    out[:, :, 3] = np.minimum(out[:, :, 3], blurred)
    return out


def match_lighting(
    overlay_bgra: np.ndarray,
    frame_bgr: np.ndarray,
    strength: float = 0.25,
) -> np.ndarray:
    """Nudge the garment's brightness toward the scene's.

    A garment shot under studio light pasted into a dim webcam frame reads as a
    sticker no matter how good the geometry is. This shifts overall luma only -
    it deliberately does not touch hue, which would destroy the garment's actual
    colour, the one thing a try-on must preserve.
    """
    if strength <= 0:
        return overlay_bgra
    alpha = overlay_bgra[:, :, 3]
    covered = alpha > 32
    if not covered.any():
        return overlay_bgra

    garment_luma = cv2.cvtColor(overlay_bgra[:, :, :3], cv2.COLOR_BGR2GRAY)[covered].mean()
    scene_luma = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean()
    if garment_luma < 1.0:
        return overlay_bgra

    gain = 1.0 + strength * (scene_luma / garment_luma - 1.0)
    gain = float(np.clip(gain, 0.6, 1.6))

    out = overlay_bgra.copy()
    out[:, :, :3] = np.clip(out[:, :, :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out
