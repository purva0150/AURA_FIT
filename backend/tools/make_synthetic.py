"""Generate a procedural placeholder garment that actually looks like a shirt.

Replaces the old generator, which baked debug anchor dots/lines directly into
the garment pixels and used a plain rectangle silhouette. That is what
produced the flat blue block with dots you saw live: warp.py and compose.py
were warping/compositing the source image correctly - the source image itself
was the problem.

This version:
    - draws a real short-sleeve tee silhouette (tapered torso, capped sleeves,
      a neckline cutout) instead of a rectangle
    - shades it (vertical light falloff + soft fold lines + fine grain) so it
      reads as fabric instead of a flat fill
    - feathers the alpha edge so the warped result doesn't look cut out
    - stores anchors ONLY in the JSON sidecar - nothing is drawn on the image
    - writes one file per angle plus a garment.json manifest, matching the
      schema GarmentSet.load()/_load_angle_json() expect

Usage (from backend/):
    python tools/make_synthetic.py --out ../garments/synthetic_tee
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np

# Angles we author directly. GarmentSet.fill_mirrors() synthesises 225/270/315
# from these by flipping, so we only need the left-turning half of the circle.
AUTHORED_ANGLES = (0, 45, 90, 135, 180)

CANVAS_W, CANVAS_H = 520, 640
BASE_COLOR_BGR = (150, 90, 40)  # muted blue-ish tone in BGR


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _view_scale(angle_deg: int) -> float:
    """How much the torso foreshortens as the view turns toward profile.

    0/180 = front/back (full width), 90 = side (narrowest).
    """
    rad = np.radians(angle_deg if angle_deg <= 180 else 360 - angle_deg)
    return float(np.clip(abs(np.cos(rad)), 0.32, 1.0))


def _build_geometry(angle_deg: int) -> Dict[str, np.ndarray]:
    """Body-relative anchor layout for one angle, in canvas pixel coords."""
    scale = _view_scale(angle_deg)

    cx = CANVAS_W / 2.0
    shoulder_w = 300.0 * scale
    hip_w = 240.0 * scale

    neck = np.array([cx, 95.0])
    left_shoulder = np.array([cx - shoulder_w / 2, 130.0])
    right_shoulder = np.array([cx + shoulder_w / 2, 130.0])
    left_hip = np.array([cx - hip_w / 2, 480.0])
    right_hip = np.array([cx + hip_w / 2, 480.0])

    # Sleeve reaches roughly to mid-upper-arm for a short-sleeve tee.
    left_elbow = np.array([cx - shoulder_w / 2 - 55.0 * scale, 270.0])
    right_elbow = np.array([cx + shoulder_w / 2 + 55.0 * scale, 270.0])

    return {
        "neck": neck,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
    }


def _tee_polygon(pts: Dict[str, np.ndarray]) -> np.ndarray:
    """Torso + sleeve silhouette as one closed polygon (hem -> hem)."""
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    lh, rh = pts["left_hip"], pts["right_hip"]
    le, re = pts["left_elbow"], pts["right_elbow"]
    neck = pts["neck"]

    sleeve_top_l = ls + (le - ls) * 0.15
    sleeve_bot_l = ls + (le - ls) * 1.05 + np.array([0, 26])
    sleeve_top_r = rs + (re - rs) * 0.15
    sleeve_bot_r = rs + (re - rs) * 1.05 + np.array([0, 26])

    poly = [
        neck + np.array([-18, -4]),
        ls,
        sleeve_top_l,
        sleeve_bot_l,
        ls + np.array([6, 30]),
        lh,
        rh,
        rs + np.array([-6, 30]),
        sleeve_bot_r,
        sleeve_top_r,
        rs,
        neck + np.array([18, -4]),
    ]
    return np.array(poly, dtype=np.float32)


def _shade(canvas: np.ndarray, alpha: np.ndarray, base_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Fabric-like shading: light-to-dark gradient + soft folds + grain."""
    h, w = alpha.shape
    base = np.array(base_bgr, dtype=np.float32)

    # Vertical falloff: slightly lighter at the shoulders, darker at the hem.
    ramp = np.linspace(1.12, 0.86, h, dtype=np.float32).reshape(h, 1)
    shaded = np.tile(base.reshape(1, 1, 3), (h, w, 1)) * ramp[:, :, None]

    # A handful of soft diagonal fold shadows, not hard lines - drawn on a
    # separate float layer and heavily blurred before blending.
    folds = np.zeros((h, w), np.float32)
    rng = np.random.default_rng(7)
    for _ in range(5):
        x0 = rng.uniform(w * 0.25, w * 0.75)
        y0 = rng.uniform(h * 0.25, h * 0.55)
        length = rng.uniform(h * 0.15, h * 0.3)
        angle = rng.uniform(-25, 25)
        dx = np.sin(np.radians(angle)) * length
        dy = np.cos(np.radians(angle)) * length
        cv2.line(
            folds,
            (int(x0 - dx / 2), int(y0 - dy / 2)),
            (int(x0 + dx / 2), int(y0 + dy / 2)),
            1.0,
            thickness=int(w * 0.03),
        )
    folds = cv2.GaussianBlur(folds, (0, 0), sigmaX=w * 0.03)
    shaded *= (1.0 - folds[:, :, None] * 0.10)

    # Fine grain so it doesn't read as a flat digital fill.
    grain = rng.normal(0, 4.0, (h, w, 1)).astype(np.float32)
    shaded += grain

    return np.clip(shaded, 0, 255).astype(np.uint8)


def render_angle(angle_deg: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    pts = _build_geometry(angle_deg)
    poly = _tee_polygon(pts)

    hard = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    cv2.fillPoly(hard, [np.round(poly).astype(np.int32)], 255)

    # Feathered alpha edge so the warped result blends instead of looking cut out.
    alpha = cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), sigmaX=2.5)
    alpha = np.clip(alpha, 0, 1)

    shaded_bgr = _shade(hard, alpha, BASE_COLOR_BGR)

    bgra = np.zeros((CANVAS_H, CANVAS_W, 4), np.uint8)
    bgra[:, :, :3] = shaded_bgr
    bgra[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return bgra, pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join("..", "garments", "synthetic_tee"))
    ap.add_argument("--name", default="Synthetic short-sleeve")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    garment_id = os.path.basename(out_dir)

    manifest_angles = []

    for angle in AUTHORED_ANGLES:
        bgra, pts = render_angle(angle)

        stem = f"{garment_id}_{angle:03d}"
        png_name = f"{stem}.png"
        json_name = f"{stem}.json"

        cv2.imwrite(os.path.join(out_dir, png_name), bgra)

        anchors = {name: [float(p[0]), float(p[1])] for name, p in pts.items()}
        payload = {
            "angle_deg": angle,
            "view": {0: "front", 45: "front_3q", 90: "side", 135: "back_3q", 180: "back"}[angle],
            "image": png_name,
            "anchors": anchors,
        }
        with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        manifest_angles.append({"anchors": json_name})
        print(f"wrote {png_name} + {json_name}")

    manifest = {
        "garment_id": garment_id,
        "name": args.name,
        "dominant_color": "#5a3c24",
        "category": "upper_body",
        "angles": manifest_angles,
    }
    with open(os.path.join(out_dir, "garment.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\ngarment.json written -> {out_dir}")
    print("Run the live app with --tps so sleeves/torso conform to the pose:")
    print("    python tryon_live.py --garment synthetic_tee --tps")


if __name__ == "__main__":
    main()