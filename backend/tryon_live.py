"""Live try-on mirror.

Supersedes ``overlay_test.py``: instead of scaling a garment by fixed multipliers
off shoulder width, this estimates body yaw, picks the two bracketing garment
angles, warps each onto measured landmark correspondences, and cross-dissolves
between them.

Also the framing aid for the guided 360 capture - you need to see your own
alignment while turning, which is why the live path exists at all.

TPS is off by default here. It costs ~75 ms a frame, which would drop the
preview to ~13 fps; the offline render pass in ``render360.py`` turns it on,
where wall-clock doesn't matter. Press ``t`` to see the difference.

Run:  python backend/tryon_live.py
      python backend/tryon_live.py --garment synthetic_tee --camera 0

Keys:  q quit   t TPS   o occlusion   s skeleton   [ ] switch garment   h help
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
    """Per-frame try-on pipeline plus its debug HUD."""

    def __init__(self, garments: List[GarmentSet], use_tps: bool = False) -> None:
        if not garments:
            raise ValueError("no garments available")
        self.garments = garments
        self.index = 0
        self.use_tps = use_tps
        self.use_occlusion = True
        self.show_skeleton = False
        self.smoother = YawSmoother(alpha=0.3)
        self._fps = deque(maxlen=30)
        self._last_residual: Optional[float] = None

    @property
    def garment(self) -> GarmentSet:
        return self.garments[self.index]

    def cycle(self, step: int) -> None:
        self.index = (self.index + step) % len(self.garments)
        self.smoother.reset()

    def process(self, frame_bgr: np.ndarray, pose_result) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if pose_result is None:
            self.smoother.update(None)
            return frame_bgr

        yaw = self.smoother.update(estimate_yaw(pose_result))
        if yaw is None:
            return frame_bgr

        lo, hi, weight = self.garment.resolve(yaw)

        overlays = []
        residuals = []
        for angle, share in ((lo, 1.0 - weight), (hi, weight)):
            if share <= 1e-3:
                overlays.append(None)
                continue
            result = warp.warp_garment(angle, pose_result, (w, h), use_tps=self.use_tps)
            overlays.append(result.image if result else None)
            if result:
                residuals.append(result.residual_px)

        overlay = compose.blend_overlays(overlays[0], overlays[1], weight)
        if overlay is None:
            return frame_bgr

        self._last_residual = float(np.mean(residuals)) if residuals else None

        if self.use_occlusion:
            mask = compose.person_occlusion_mask(pose_result, (w, h))
            overlay = compose.apply_occlusion(overlay, mask)

        overlay = compose.feather_alpha(overlay, radius=2)
        overlay = compose.match_lighting(overlay, frame_bgr)
        out = compose.alpha_composite(frame_bgr, overlay)

        if self.show_skeleton:
            draw_skeleton(out, pose_result)
        return out

    def draw_hud(self, frame: np.ndarray, pose_result) -> np.ndarray:
        h, w = frame.shape[:2]
        yaw = self.smoother.value

        cv2.rectangle(frame, (0, 0), (w, 96), _HUD_BG, -1)
        cv2.putText(frame, self.garment.name, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _HUD_FG, 2)

        if yaw is None:
            cv2.putText(frame, "no pose", (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 220), 2)
        else:
            bin_deg = nearest_bin(yaw)
            txt = f"yaw {yaw:6.1f}deg   bin {int(bin_deg):3d}   stability {self.smoother.stability:.2f}"
            cv2.putText(frame, txt, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _HUD_FG, 1)

        flags = f"TPS {'on' if self.use_tps else 'off'}   occl {'on' if self.use_occlusion else 'off'}"
        if self._last_residual is not None:
            flags += f"   residual {self._last_residual:.1f}px"
        if self._fps:
            flags += f"   {np.mean(self._fps):.0f} fps"
        cv2.putText(frame, flags, (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)

        if yaw is not None:
            self._draw_dial(frame, yaw, (w - 80, 60), 40)
        return frame

    def _draw_dial(self, frame: np.ndarray, yaw: float, center, radius: int) -> None:
        """Compass showing current yaw against the available garment angles.

        Filled ticks are angles the garment actually has; hollow ones are gaps
        being covered by cross-dissolve. Makes a three-angle catalogue garment
        visibly different from a fully photographed one.
        """
        cv2.circle(frame, center, radius, (90, 90, 90), 1)
        available = set(self.garment.available_angles)
        for b in CAPTURE_BINS:
            ang = np.radians(b - 90)
            p = (int(center[0] + np.cos(ang) * radius), int(center[1] + np.sin(ang) * radius))
            filled = any(abs((a - b + 180) % 360 - 180) < 5 for a in available)
            cv2.circle(frame, p, 3, _ACCENT if filled else (110, 110, 110), -1 if filled else 1)
        ang = np.radians(yaw - 90)
        tip = (int(center[0] + np.cos(ang) * (radius - 8)), int(center[1] + np.sin(ang) * (radius - 8)))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--garment", default=None, help="garment id or path (default: all found)")
    ap.add_argument("--garments-dir", default=GARMENTS_DIR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tps", action="store_true", help="start with TPS enabled (slow)")
    ap.add_argument("--no-mirror-view", action="store_true", help="don't flip the preview horizontally")
    args = ap.parse_args()

    garments = load_garments(args.garments_dir, args.garment)
    for g in garments:
        gaps = g.coverage_gaps(CAPTURE_BINS)
        if gaps:
            print(f"note: '{g.garment_id}' has no asset near {[int(x) for x in gaps]} deg "
                  f"- those angles will be covered by cross-dissolve")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    app = LiveTryOn(garments, use_tps=args.tps)
    window = "Virtual Try-On - live"
    print(__doc__.split("Keys:")[-1].strip())

    last = time.perf_counter()
    with PoseEstimator(segmentation=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if not args.no_mirror_view:
                # Mirror so it reads as a mirror. Done before pose so landmark
                # left/right stay consistent with what the user sees.
                frame = cv2.flip(frame, 1)

            result = pose.process(frame)
            out = app.process(frame, result)
            app.draw_hud(out, result)

            now = time.perf_counter()
            app.tick(now - last)
            last = now

            cv2.imshow(window, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("t"):
                app.use_tps = not app.use_tps
            elif key == ord("o"):
                app.use_occlusion = not app.use_occlusion
            elif key == ord("s"):
                app.show_skeleton = not app.show_skeleton
            elif key == ord("]"):
                app.cycle(1)
            elif key == ord("["):
                app.cycle(-1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
