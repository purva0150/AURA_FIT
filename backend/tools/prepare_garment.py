"""Prepare a catalogue garment image for the live try-on pipeline.

The common failure case is a PNG that technically has an alpha channel but
still contains an opaque white/solid-colour background.  A global colour-key
would also erase a white T-shirt, so this tool removes only background-like
pixels connected to the image border.

Example (run from the repository root):
    python backend/tools/prepare_garment.py "white tshirt.png" \
        --id white_oversized_tee --name "White oversized T-shirt"
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def border_background_alpha(image: Image.Image, tolerance: float = 30.0) -> np.ndarray:
    """Return alpha after removing border-connected background pixels."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.int16)
    height, width = rgb.shape[:2]

    corners = np.array(
        [rgb[0, 0], rgb[0, width - 1], rgb[height - 1, 0], rgb[height - 1, width - 1]],
        dtype=np.float32,
    )
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    candidate = distance <= float(tolerance)

    connected = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not candidate[y, x]:
            continue
        connected[y, x] = True
        if y: queue.append((y - 1, x))
        if y + 1 < height: queue.append((y + 1, x))
        if x: queue.append((y, x - 1))
        if x + 1 < width: queue.append((y, x + 1))

    alpha = rgba[:, :, 3].copy()
    alpha[connected] = 0
    # A tiny blur gives anti-aliased catalogue edges without a glowing halo.
    return np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.65)))


def default_anchors(width: int, height: int) -> dict[str, list[float]]:
    """Conservative anchors for a centred, front-facing short-sleeve top."""
    return {
        # Anatomical neck/shoulder midpoint, not the visible collar edge.
        "neck": [0.500 * width, 0.255 * height],
        # MediaPipe names anatomical sides: the wearer's left appears on the
        # right side of a front-facing catalogue image.
        "left_shoulder": [0.720 * width, 0.255 * height],
        "right_shoulder": [0.280 * width, 0.255 * height],
        "left_elbow": [0.860 * width, 0.535 * height],
        "right_elbow": [0.110 * width, 0.535 * height],
        "left_hip": [0.685 * width, 0.875 * height],
        "right_hip": [0.295 * width, 0.875 * height],
        "left_chest": [0.690 * width, 0.450 * height],
        "right_chest": [0.310 * width, 0.450 * height],
        "left_waist": [0.680 * width, 0.670 * height],
        "right_waist": [0.320 * width, 0.670 * height],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--id", required=True, dest="garment_id")
    parser.add_argument("--name", required=True)
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--garments-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    garments_dir = Path(args.garments_dir) if args.garments_dir else repo_root / "garments"
    output_dir = garments_dir / args.garment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGBA")
    rgba = np.asarray(image).copy()
    rgba[:, :, 3] = border_background_alpha(image, args.tolerance)
    prepared = Image.fromarray(rgba, mode="RGBA")

    # Catalogue photos often contain the dark inside/back of the collar as
    # opaque pixels.  That region must reveal the wearer's neck.  Cut only the
    # conservative inner opening; the visible ribbed collar stays intact.
    alpha_image = prepared.getchannel("A")
    draw = ImageDraw.Draw(alpha_image)
    width, height = prepared.size
    draw.ellipse(
        (0.425 * width, 0.098 * height, 0.575 * width, 0.142 * height),
        fill=0,
    )
    prepared.putalpha(alpha_image.filter(ImageFilter.GaussianBlur(0.45)))

    image_name = f"{args.garment_id}_000.png"
    anchors_name = f"{args.garment_id}_000.json"
    prepared.save(output_dir / image_name)

    anchors = {
        "garment_id": args.garment_id,
        "angle_deg": 0,
        "view": "front",
        "image": image_name,
        "anchors": default_anchors(*prepared.size),
        "fit": {
            "hem_extension": 0.10,
            "hem_width_ratio": 0.92,
            "sleeve_length_ratio": 0.52,
            "chest_width_ratio": 0.94,
            "waist_width_ratio": 0.86
        },
    }
    (output_dir / anchors_name).write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    manifest = {
        "garment_id": args.garment_id,
        "name": args.name,
        "dominant_color": "#eeeeee",
        "category": "upper_body",
        "angles": [{"anchors": anchors_name}],
    }
    (output_dir / "garment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared garment: {output_dir}")


if __name__ == "__main__":
    main()
