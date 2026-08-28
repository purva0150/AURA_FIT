"""Generate a procedural multi-angle garment so the pipeline is testable today.

Real assets need photography or a dataset download; neither should block the
warp, capture and viewer code. This renders a parametric tee at all eight yaw
bins with anchor points that are **exact by construction** - they come from the
same projection maths that draws the pixels, so warp residuals measured against
them are real error rather than annotation slop.

Projection model
----------------
Body frame axes: ``u`` is the person's left, ``w`` is their forward. Rotating by
yaw ``t`` (same convention as ``tryon.yaw``: 0 = facing camera) projects to::

    image_x = u*cos(t) + w*sin(t)
    depth_z = u*sin(t) - w*cos(t)          # smaller is nearer, as in MediaPipe

The torso is an elliptical cylinder with semi-axes ``a`` across and ``b`` deep,
so its projected half-width is ``sqrt((a*cos t)^2 + (b*sin t)^2)`` - full width
head-on, narrowing to the body's depth at 90 degrees. That single relation is
what makes the silhouette read as a rotating solid instead of a flat card.

Per-column Lambertian shading uses the true ellipse normal, so the highlight
travels around the body as it turns. Cheap, and it makes bin transitions
visually obvious when debugging the blend.

Run:  python backend/tools/make_synthetic.py
      python backend/tools/make_synthetic.py --id navy_tee --color 2f5fa8 --sleeve long
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tryon.yaw import CAPTURE_BINS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GARMENTS_DIR = os.path.join(REPO_ROOT, "garments")

CANVAS_W, CANVAS_H = 512, 700

# Light direction in camera space (x right, z depth with negative toward the
# camera): upper-left and in front of the subject.
_LIGHT = np.array([-0.45, -0.89])
_AMBIENT = 0.55

VIEW_NAMES: Dict[int, str] = {
    0: "front",
    45: "front45",
    90: "side",
    135: "back45",
    180: "back",
    225: "back45_mirror",
    270: "side_mirror",
    315: "front45_mirror",
}


@dataclass
class GarmentSpec:
    """Parametric description of one garment, in canvas pixels."""

    garment_id: str = "synthetic_tee"
    name: str = "Synthetic Tee (procedural)"
    color: Tuple[int, int, int] = (168, 95, 47)  # BGR
    shoulder_y: int = 150
    hip_y: int = 540
    hem_y: int = 570
    a_shoulder: float = 108.0  # half shoulder width
    b_shoulder: float = 60.0   # half torso depth at the shoulders
    a_hip: float = 95.0
    b_hip: float = 57.0
    sleeve: str = "short"      # "short" | "long" | "sleeveless"
    collar_drop: int = 26
    trim_color: Tuple[int, int, int] = (210, 210, 210)
    angles: Tuple[float, ...] = field(default_factory=lambda: CAPTURE_BINS)

    @property
    def sleeve_reach(self) -> float:
        return {"sleeveless": 0.10, "short": 0.62, "long": 1.35}[self.sleeve]

    @property
    def sleeve_drop(self) -> float:
        return {"sleeveless": 30.0, "short": 132.0, "long": 250.0}[self.sleeve]

    @property
    def sleeve_radius(self) -> float:
        return {"sleeveless": 34.0, "short": 40.0, "long": 32.0}[self.sleeve]


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _axes_at(spec: GarmentSpec, y: float) -> Tuple[float, float]:
    """Ellipse semi-axes at height ``y``, tapering shoulders to hips."""
    t = np.clip((y - spec.shoulder_y) / max(spec.hip_y - spec.shoulder_y, 1), 0.0, 1.4)
    return _lerp(spec.a_shoulder, spec.a_hip, t), _lerp(spec.b_shoulder, spec.b_hip, t)


def _half_width(a: float, b: float, theta: float) -> float:
    return math.hypot(a * math.cos(theta), b * math.sin(theta))


def _column_shade(dx: np.ndarray, a: float, b: float, theta: float) -> np.ndarray:
    """Lambertian shade per column, from the real ellipse normal.

    For a column offset ``dx`` there are two ellipse points that project there -
    one on the near surface, one on the far. We keep whichever normal points at
    the camera (more negative z).
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hw = _half_width(a, b, theta)
    if hw < 1e-6:
        return np.full_like(dx, _AMBIENT, dtype=np.float64)

    psi = math.atan2(b * sin_t, a * cos_t)
    delta = np.arccos(np.clip(dx / hw, -1.0, 1.0))

    best_nx = np.zeros_like(dx, dtype=np.float64)
    best_nz = np.full_like(dx, np.inf, dtype=np.float64)
    for phi in (psi + delta, psi - delta):
        nu, nw = np.cos(phi) / a, np.sin(phi) / b
        norm = np.hypot(nu, nw)
        norm[norm < 1e-9] = 1e-9
        nu, nw = nu / norm, nw / norm
        nx = nu * cos_t + nw * sin_t
        nz = nu * sin_t - nw * cos_t
        take = nz < best_nz
        best_nx = np.where(take, nx, best_nx)
        best_nz = np.where(take, nz, best_nz)

    lambert = np.clip(best_nx * _LIGHT[0] + best_nz * _LIGHT[1], 0.0, 1.0)
    return _AMBIENT + (1.0 - _AMBIENT) * lambert


def _shoulder_points(spec: GarmentSpec, theta: float) -> Dict[str, Tuple[float, float, float]]:
    """Projected shoulder/hip joints as (x, y, depth)."""
    cx = CANVAS_W / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return {
        "left_shoulder": (cx + spec.a_shoulder * cos_t, spec.shoulder_y, spec.a_shoulder * sin_t),
        "right_shoulder": (cx - spec.a_shoulder * cos_t, spec.shoulder_y, -spec.a_shoulder * sin_t),
        "left_hip": (cx + spec.a_hip * cos_t, spec.hip_y, spec.a_hip * sin_t),
        "right_hip": (cx - spec.a_hip * cos_t, spec.hip_y, -spec.a_hip * sin_t),
    }


def _elbow_points(spec: GarmentSpec, theta: float) -> Dict[str, Tuple[float, float, float]]:
    cx = CANVAS_W / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = spec.a_shoulder * (1.0 + spec.sleeve_reach)
    y = spec.shoulder_y + spec.sleeve_drop
    return {
        "left_elbow": (cx + u * cos_t, y, u * sin_t),
        "right_elbow": (cx - u * cos_t, y, -u * sin_t),
    }


def _draw_torso(canvas: np.ndarray, alpha: np.ndarray, spec: GarmentSpec, theta: float) -> None:
    cx = CANVAS_W / 2.0
    for y in range(spec.shoulder_y - spec.collar_drop, spec.hem_y + 1):
        a, b = _axes_at(spec, y)

        # Shoulders round off above the shoulder line rather than ending square.
        if y < spec.shoulder_y:
            k = (spec.shoulder_y - y) / max(spec.collar_drop, 1)
            a *= math.sqrt(max(0.0, 1.0 - k * k))
            b *= math.sqrt(max(0.0, 1.0 - k * k))

        hw = _half_width(a, b, theta)
        if hw < 1.0:
            continue

        x0, x1 = int(round(cx - hw)), int(round(cx + hw))
        x0, x1 = max(x0, 0), min(x1, CANVAS_W - 1)
        if x1 <= x0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        shade = _column_shade(xs - cx, a, b, theta)

        # Slight darkening toward the hem so the drape reads.
        depth_fade = 1.0 - 0.12 * np.clip((y - spec.shoulder_y) / max(spec.hem_y - spec.shoulder_y, 1), 0, 1)
        rgb = np.array(spec.color, dtype=np.float64)[None, :] * (shade * depth_fade)[:, None]
        canvas[y, x0 : x1 + 1] = np.clip(rgb, 0, 255).astype(np.uint8)
        alpha[y, x0 : x1 + 1] = 255


def _draw_sleeve(
    canvas: np.ndarray,
    alpha: np.ndarray,
    spec: GarmentSpec,
    theta: float,
    side: str,
) -> None:
    if spec.sleeve == "sleeveless":
        return
    cx = CANVAS_W / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    sign = 1.0 if side == "left" else -1.0

    root_x = cx + sign * spec.a_shoulder * 0.85 * cos_t
    root_y = float(spec.shoulder_y + 6)
    tip_x = cx + sign * spec.a_shoulder * (1.0 + spec.sleeve_reach) * cos_t
    tip_y = float(spec.shoulder_y + spec.sleeve_drop)

    # Outward-facing normal of the sleeve tube is +/- the body-left axis.
    nx, nz = sign * cos_t, sign * sin_t
    lambert = max(0.0, nx * _LIGHT[0] + nz * _LIGHT[1])
    shade = _AMBIENT + (1.0 - _AMBIENT) * lambert
    color = tuple(float(np.clip(c * shade, 0, 255)) for c in spec.color)

    r = int(spec.sleeve_radius)
    p0 = (int(round(root_x)), int(round(root_y)))
    p1 = (int(round(tip_x)), int(round(tip_y)))

    for target, val in ((canvas, color), (alpha, 255)):
        cv2.line(target, p0, p1, val, thickness=2 * r, lineType=cv2.LINE_AA)
        cv2.circle(target, p0, r, val, -1, lineType=cv2.LINE_AA)
        cv2.circle(target, p1, r, val, -1, lineType=cv2.LINE_AA)


def _draw_details(canvas: np.ndarray, alpha: np.ndarray, spec: GarmentSpec, theta: float) -> None:
    """Front placket and back yoke - orientation cues so a wrong-angle asset is
    obvious at a glance instead of silently looking plausible."""
    cx = CANVAS_W / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a_s, b_s = spec.a_shoulder, spec.b_shoulder

    if cos_t > 0.08:  # front surface visible
        x = int(round(cx + b_s * sin_t))
        y0, y1 = spec.shoulder_y + 10, spec.hip_y - 40
        cv2.line(canvas, (x, y0), (x, y1), spec.trim_color, 3, cv2.LINE_AA)
        for yy in range(y0 + 30, y1, 70):
            cv2.circle(canvas, (x, yy), 4, spec.trim_color, -1, cv2.LINE_AA)

    if cos_t < -0.08:  # back surface visible
        x = int(round(cx - b_s * sin_t))
        y = spec.shoulder_y + 46
        hw = _half_width(a_s, b_s, theta)
        cv2.line(
            canvas,
            (int(round(x - hw * 0.55)), y),
            (int(round(x + hw * 0.55)), y),
            spec.trim_color,
            2,
            cv2.LINE_AA,
        )

    # Collar ring sits on the shoulder line, squashing as the body turns.
    hw_neck = _half_width(a_s * 0.34, b_s * 0.5, theta)
    cv2.ellipse(
        canvas,
        (int(cx), spec.shoulder_y),
        (max(int(hw_neck), 2), 14),
        0,
        0,
        360,
        spec.trim_color,
        3,
        cv2.LINE_AA,
    )


def render_angle(spec: GarmentSpec, angle_deg: float) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Render one yaw angle. Returns (BGRA image, anchor dict)."""
    theta = math.radians(angle_deg)
    # uint8 throughout: OpenCV's drawing primitives reject float64 buffers.
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    alpha = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    joints = _shoulder_points(spec, theta)
    elbows = _elbow_points(spec, theta)

    # Painter's algorithm: the sleeve behind the torso goes down first.
    sleeves = sorted(("left", "right"), key=lambda s: -elbows[f"{s}_elbow"][2])
    _draw_sleeve(canvas, alpha, spec, theta, sleeves[0])
    _draw_torso(canvas, alpha, spec, theta)
    _draw_details(canvas, alpha, spec, theta)
    _draw_sleeve(canvas, alpha, spec, theta, sleeves[1])

    bgra = np.dstack([canvas, alpha])

    # Feather the silhouette so compositing doesn't show a hard cut edge.
    blurred = cv2.GaussianBlur(bgra[:, :, 3], (5, 5), 0)
    bgra[:, :, 3] = np.minimum(bgra[:, :, 3], blurred)

    anchors: Dict[str, List[float]] = {}
    for name, (x, y, _z) in {**joints, **elbows}.items():
        anchors[name] = [round(float(x), 2), round(float(y), 2)]
    cx = CANVAS_W / 2.0
    anchors["neck"] = [round(cx, 2), round(float(spec.shoulder_y), 2)]
    return bgra, anchors


def _view_name(angle_deg: float) -> str:
    return VIEW_NAMES.get(int(round(angle_deg)), f"deg{int(round(angle_deg))}")


def generate(spec: GarmentSpec, out_root: str = GARMENTS_DIR) -> str:
    out_dir = os.path.join(out_root, spec.garment_id)
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    for angle in spec.angles:
        bgra, anchors = render_angle(spec, angle)
        deg = int(round(angle)) % 360
        img_name = f"{spec.garment_id}_{deg:03d}.png"
        json_name = f"{spec.garment_id}_{deg:03d}.json"

        cv2.imwrite(os.path.join(out_dir, img_name), bgra)
        payload = {
            "garment_id": spec.garment_id,
            "angle_deg": deg,
            "view": _view_name(angle),
            "image": img_name,
            "source": {"kind": "synthetic", "generator": "backend/tools/make_synthetic.py"},
            "anchors": anchors,
        }
        with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        entries.append({"angle_deg": deg, "view": _view_name(angle), "image": img_name, "anchors": json_name})
        print(f"  {deg:>3}deg  {img_name}")

    b, g, r = spec.color
    manifest = {
        "garment_id": spec.garment_id,
        "name": spec.name,
        "category": "upper_body",
        "dominant_color": f"#{r:02x}{g:02x}{b:02x}",
        "sleeve": spec.sleeve,
        "source": {"kind": "synthetic", "generator": "backend/tools/make_synthetic.py"},
        "angles": entries,
    }
    manifest_path = os.path.join(out_dir, "garment.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out_dir


def _parse_color(text: str) -> Tuple[int, int, int]:
    text = text.lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError("colour must be 6 hex digits, e.g. 2f5fa8")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", default="synthetic_tee", help="garment id / folder name")
    ap.add_argument("--name", default=None, help="display name")
    ap.add_argument("--color", type=_parse_color, default="2f5fa8", help="hex RGB, e.g. 2f5fa8")
    ap.add_argument("--sleeve", choices=("sleeveless", "short", "long"), default="short")
    ap.add_argument("--out", default=GARMENTS_DIR)
    args = ap.parse_args()

    color = args.color if isinstance(args.color, tuple) else _parse_color(args.color)
    spec = GarmentSpec(
        garment_id=args.id,
        name=args.name or f"Synthetic {args.sleeve}-sleeve ({args.id})",
        color=color,
        sleeve=args.sleeve,
    )
    print(f"Rendering '{spec.garment_id}' at {len(spec.angles)} angles...")
    out_dir = generate(spec, args.out)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
