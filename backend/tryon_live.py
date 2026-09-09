"""Live virtual try-on mirror.

Run from backend:
    python tryon_live.py --garment synthetic_tee

Keys: q/ESC quit | t TPS | o arm occlusion | s skeleton | [ ] garment | h help
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tryon import compose, warp  # noqa: E402
from tryon.garment import GarmentSet, discover_garments  # noqa: E402
from tryon.pose import PoseEstimator, draw_skeleton  # noqa: E402
from tryon.yaw import CAPTURE_BINS, YawSmoother, estimate_yaw, nearest_bin  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GARMENTS_DIR = os.path.join(REPO_ROOT, "garments")

_HUD_BG = (28, 28, 28)
_HUD_FG = (240, 240, 240)
_ACCENT = (168, 95, 47)


class LiveTryOn:
    def __init__(self, garments: List[GarmentSet], use_tps: bool = True) -> None:
        # NOTE: use_tps now defaults to True. With it off, the garment only
        # gets a rigid similarity transform (rotate/scale/translate as one
        # flat piece) - no sleeve/torso conformance to the actual pose. That
        # rigid-only path is what produced the blocky, non-fitted look.
        # TPS costs FPS (see LiveTryOn.tick / HUD), so still exposed via 't'.
        if not garments:
            raise ValueError("no garments available")
        self.garments = garments
        self.index = 0
        self.use_tps = use_tps
        self.use_occlusion = True
        self.show_skeleton = False
        self.smoother = YawSmoother(alpha=0.30)
        self._fps = deque(maxlen=30)
        self._last_residual: Optional[float] = None

    @property
    def garment(self) -> GarmentSet:
        return self.garments[self.index]

    def cycle(self, step: int) -> None:
        self.index = (self.index + step) % len(self.garments)
        self.smoother.reset()

    def process(self, frame_bgr: np.ndarray, pose_result, view_mode: str = "auto", fit_scale: float = 1.0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if pose_result is None:
            self.smoother.update(None)
            return frame_bgr

        tracked_yaw = self.smoother.update(estimate_yaw(pose_result))
        yaw = 0.0 if view_mode == "front" else 180.0 if view_mode == "back" else tracked_yaw
        if yaw is None:
            return frame_bgr

        lo, hi, weight = self.garment.resolve(yaw)

        overlays = []
        residuals = []
        for angle, share in ((lo, 1.0 - weight), (hi, weight)):
            if share <= 1e-3:
                overlays.append(None)
                continue
            result = warp.warp_garment(
                angle,
                pose_result,
                (w, h),
                use_tps=self.use_tps,
                fit_scale=fit_scale,
            )
            overlays.append(result.image if result else None)
            if result is not None:
                residuals.append(result.residual_px)

        overlay = compose.blend_overlays(overlays[0], overlays[1], weight)
        if overlay is None:
            return frame_bgr

        self._last_residual = float(np.mean(residuals)) if residuals else None

        # =============================================================
        # 1. REPLACE THE OLD SHIRT
        # =============================================================
        # This is deliberately NOT the arm occlusion mask.
        # It is a body-following replacement region. The compositor keeps
        # the new garment opaque inside it, so the old shirt cannot show
        # through as it did with the previous soft-alpha implementation.
        clothing_mask = compose.torso_clothing_mask(
            pose_result,
            (w, h),
        )

        # Do not clip the new garment to this approximate polygon. The asset's
        # alpha is the authoritative shirt silhouette; using the body mask as
        # scissors creates flat collars and chopped hems. The opaque warped
        # garment covers the old shirt, while the mask remains available for
        # future semantic inpainting/offline VTON.

        # =============================================================
        # 2. PUT REAL FOREARMS/HANDS BACK IN FRONT
        # =============================================================
        if self.use_occlusion:
            occlusion_mask = compose.person_occlusion_mask(
                pose_result,
                (w, h),
                use_depth_gate=True,
            )
            overlay = compose.apply_occlusion(
                overlay,
                occlusion_mask,
            )

        # =============================================================
        # 3. EDGE + LIGHTING POLISH
        # =============================================================
        overlay = compose.feather_alpha(overlay, radius=1)
        overlay = compose.match_lighting(
            overlay,
            frame_bgr,
            strength=0.10,
        )

        replacement_base = compose.skin_tone_underlay(
            frame_bgr,
            pose_result,
            clothing_mask,
            overlay,
        )
        out = compose.alpha_composite(replacement_base, overlay)

        if self.show_skeleton:
            draw_skeleton(out, pose_result)

        return out

    def draw_hud(self, frame: np.ndarray, pose_result) -> np.ndarray:
        h, w = frame.shape[:2]
        yaw = self.smoother.value

        cv2.rectangle(frame, (0, 0), (w, 96), _HUD_BG, -1)
        cv2.putText(
            frame,
            self.garment.name,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _HUD_FG,
            2,
        )

        if yaw is None:
            cv2.putText(
                frame,
                "no pose",
                (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 80, 220),
                2,
            )
        else:
            bin_deg = nearest_bin(yaw)
            txt = (
                f"yaw {yaw:6.1f}deg   bin {int(bin_deg):3d}   "
                f"stability {self.smoother.stability:.2f}"
            )
            cv2.putText(
                frame,
                txt,
                (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _HUD_FG,
                1,
            )

        flags = (
            f"TPS {'on' if self.use_tps else 'off'}   "
            f"occl {'on' if self.use_occlusion else 'off'}"
        )
        if self._last_residual is not None:
            flags += f"   residual {self._last_residual:.1f}px"
        if self._fps:
            flags += f"   {np.mean(self._fps):.0f} fps"

        cv2.putText(
            frame,
            flags,
            (14, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (170, 170, 170),
            1,
        )

        if yaw is not None:
            self._draw_dial(frame, yaw, (w - 80, 60), 40)
        return frame

    def _draw_dial(self, frame, yaw, center, radius: int) -> None:
        cv2.circle(frame, center, radius, (90, 90, 90), 1)
        available = set(self.garment.available_angles)

        for b in CAPTURE_BINS:
            ang = np.radians(b - 90)
            p = (
                int(center[0] + np.cos(ang) * radius),
                int(center[1] + np.sin(ang) * radius),
            )
            filled = any(
                abs((a - b + 180) % 360 - 180) < 5
                for a in available
            )
            cv2.circle(
                frame,
                p,
                3,
                _ACCENT if filled else (110, 110, 110),
                -1 if filled else 1,
            )

        ang = np.radians(yaw - 90)
        tip = (
            int(center[0] + np.cos(ang) * (radius - 8)),
            int(center[1] + np.sin(ang) * (radius - 8)),
        )
        cv2.line(frame, center, tip, (80, 220, 120), 2)

    def tick(self, dt: float) -> None:
        if dt > 0:
            self._fps.append(1.0 / dt)


def load_garments(garments_dir: str, only: Optional[str]) -> List[GarmentSet]:
    if only:
        path = only if os.path.isdir(only) else os.path.join(garments_dir, only)
        return [GarmentSet.load(path)]

    found = discover_garments(garments_dir)
    if not found:
        raise SystemExit(
            f"No garments found in {garments_dir}.\n"
            "Generate the procedural placeholder first:\n"
            "    python backend/tools/make_synthetic.py"
        )
    return found


def print_help() -> None:
    print("\nVirtual Try-On Controls")
    print("=" * 60)
    print("q / ESC   Quit")
    print("t         Toggle TPS")
    print("o         Toggle body/arm occlusion")
    print("s         Toggle skeleton")
    print("[         Previous garment")
    print("]         Next garment")
    print("h         Show this help")
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--garment", default=None)
    ap.add_argument("--garments-dir", default=GARMENTS_DIR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument(
        "--no-tps",
        action="store_true",
        help="Disable Thin Plate Spline refinement (rigid similarity only, higher FPS).",
    )
    ap.add_argument("--no-mirror-view", action="store_true")
    args = ap.parse_args()

    garments = load_garments(args.garments_dir, args.garment)
    for g in garments:
        gaps = g.coverage_gaps(CAPTURE_BINS)
        if gaps:
            print(
                f"note: '{g.garment_id}' has no asset near "
                f"{[int(x) for x in gaps]} deg - cross-dissolve will cover gaps"
            )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    app = LiveTryOn(garments, use_tps=not args.no_tps)
    window = "Virtual Try-On - live"

    print("\nVirtual Try-On started.")
    print_help()

    last = time.perf_counter()

    with PoseEstimator(segmentation=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if not args.no_mirror_view:
                frame = cv2.flip(frame, 1)

            result = pose.process(frame)
            out = app.process(frame, result)
            out = app.draw_hud(out, result)

            now = time.perf_counter()
            app.tick(now - last)
            last = now

            cv2.imshow(window, out)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord("t"):
                app.use_tps = not app.use_tps
            elif key == ord("o"):
                app.use_occlusion = not app.use_occlusion
            elif key == ord("s"):
                app.show_skeleton = not app.show_skeleton
            elif key == ord("]"):
                app.cycle(1)
            elif key == ord("["):
                app.cycle(-1)
            elif key == ord("h"):
                print_help()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
