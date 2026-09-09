"""Render one still image through the same pipeline used by the webcam.

Example:
    python backend/render_tryon_image.py person.png \
        --garment white_oversized_tee --output tryon_preview.png
"""
from __future__ import annotations

import argparse
import os

import cv2

from tryon.garment import GarmentSet
from tryon.pose import PoseEstimator
from tryon_live import GARMENTS_DIR, LiveTryOn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--garment", required=True)
    parser.add_argument("--output", default="tryon_preview.png")
    parser.add_argument("--show-skeleton", action="store_true")
    parser.add_argument("--no-tps", action="store_true")
    args = parser.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read input image: {args.image}")

    garment_path = args.garment
    if not os.path.isdir(garment_path):
        garment_path = os.path.join(GARMENTS_DIR, garment_path)
    garment = GarmentSet.load(garment_path)

    app = LiveTryOn([garment], use_tps=not args.no_tps)
    app.show_skeleton = args.show_skeleton
    with PoseEstimator(segmentation=True, model_complexity=2, static=True) as estimator:
        pose = estimator.process(frame)
    if pose is None:
        raise SystemExit(
            "No body pose was detected. Use a brighter image that shows both "
            "shoulders and most of the torso."
        )

    output = app.process(frame, pose)
    if not cv2.imwrite(args.output, output):
        raise SystemExit(f"Could not write output image: {args.output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
