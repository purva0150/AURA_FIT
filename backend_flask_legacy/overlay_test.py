"""
Milestone 2: Garment overlay (simple alignment, no warping yet)

Takes the shoulder/hip keypoints from pose detection and scales + positions
a garment PNG (with transparency) over the torso. This is intentionally
crude - no TPS warping, no rotation handling - just to prove the
pose -> garment pipeline connects end to end before we invest in Phase 9's
warping work.

Run: python overlay_test.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

import os
from PIL import Image

GARMENT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'garments', 'shirt1_placeholder.png'))


def overlay_transparent(background, overlay, x, y, overlay_w, overlay_h):
    """Alpha-blend `overlay` (RGBA) onto `background` (BGR) at (x, y)."""
    if overlay_w <= 0 or overlay_h <= 0:
        return background

    overlay_resized = cv2.resize(overlay, (overlay_w, overlay_h))

    h, w = background.shape[:2]
    # Clip to frame bounds so we don't crash when the garment goes off-screen
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + overlay_w, w), min(y + overlay_h, h)
    if x1 >= x2 or y1 >= y2:
        return background

    overlay_crop = overlay_resized[y1 - y : y2 - y, x1 - x : x2 - x]
    alpha = overlay_crop[:, :, 3:4] / 255.0
    bg_region = background[y1:y2, x1:x2]
    blended = (alpha * overlay_crop[:, :, :3] + (1 - alpha) * bg_region).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background


def main():
    garment = cv2.imread(GARMENT_PATH, cv2.IMREAD_UNCHANGED)
    if garment is None:
        # Try PIL fallback in case OpenCV couldn't read the PNG directly
        try:
            pil_img = Image.open(GARMENT_PATH).convert('RGBA')
            arr = np.array(pil_img)
            # Convert RGBA (PIL) to BGRA (OpenCV)
            garment = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            print(f"Loaded garment via PIL fallback: {GARMENT_PATH}")
        except Exception as e:
            print(f"ERROR: couldn't load garment image at {GARMENT_PATH} ({e})")
            return
    if garment.shape[2] != 4:
        print("ERROR: garment image needs an alpha channel (transparent PNG)")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                # Convert normalized landmarks to pixel coords
                l_sh_px = (int(l_sh.x * frame_w), int(l_sh.y * frame_h))
                r_sh_px = (int(r_sh.x * frame_w), int(r_sh.y * frame_h))

                # Use hip landmarks only when they're reliable; MediaPipe can estimate
                # hips off-frame which makes torso height wrong. Fall back to a
                # shoulder-based heuristic when hips aren't confident.
                hip_visible_score = 0.0
                try:
                    hip_visible_score = (getattr(l_hip, 'visibility', 0.0) + getattr(r_hip, 'visibility', 0.0)) / 2.0
                except Exception:
                    hip_visible_score = 0.0

                # Safe pixel conversions (only if landmark values are in [0,1])
                def in_frame(y):
                    return (y is not None) and (0.0 <= y <= 1.0)

                shoulder_width_px = abs(l_sh_px[0] - r_sh_px[0])

                if hip_visible_score > 0.45 and in_frame(l_hip.y) and in_frame(r_hip.y):
                    hip_y_px = int(((l_hip.y + r_hip.y) / 2) * frame_h)
                    torso_height_px = max(1, abs(hip_y_px - int((l_sh.y + r_sh.y) / 2 * frame_h)))
                else:
                    # Fallback: estimate torso height from shoulder width (conservative)
                    # and from shoulder-to-nose vertical distance when available.
                    torso_est_from_width = int(shoulder_width_px * 1.0)
                    try:
                        nose = lm[mp_pose.PoseLandmark.NOSE]
                        nose_y_px = int(nose.y * frame_h)
                        shoulder_mid_y = int(((l_sh.y + r_sh.y) / 2) * frame_h)
                        shoulder_to_nose = abs(shoulder_mid_y - nose_y_px)
                        # Combine heuristics (at least one should be nonzero)
                        torso_height_px = max(1, int((torso_est_from_width * 0.5) + (shoulder_to_nose * 1.0)))
                    except Exception:
                        torso_height_px = max(1, torso_est_from_width)

                # Conservative scale factors to avoid excessively wide/long placeholders
                WIDTH_SCALE = 1.25  # smaller overhang than before
                HEIGHT_SCALE = 1.15

                garment_w = int(shoulder_width_px * WIDTH_SCALE)
                garment_h = int(torso_height_px * HEIGHT_SCALE)

                # Clamp sizes to frame so overlay doesn't grow off-screen
                garment_w = max(1, min(garment_w, int(frame_w * 0.95)))
                garment_h = max(1, min(garment_h, int(frame_h * 0.9)))

                # Top-left corner: center horizontally on shoulder midpoint,
                # start slightly above the shoulder line. Then enforce it sits below
                # the face (avoid overlapping chin) and inside the frame.
                mid_x = (l_sh_px[0] + r_sh_px[0]) // 2
                top_y = min(l_sh_px[1], r_sh_px[1]) - int(garment_h * 0.05)
                top_x = mid_x - garment_w // 2

                # If nose is available, use it to prevent overlapping the chin/face
                try:
                    nose = lm[mp_pose.PoseLandmark.NOSE]
                    nose_y_px = int(nose.y * frame_h)
                    # require top edge to be below the nose + small margin (so it doesn't overlap chin)
                    min_top = nose_y_px + max(4, int(frame_h * 0.02))
                    if top_y < min_top:
                        top_y = min_top
                except Exception:
                    pass

                # Ensure top_x and top_y are within the frame; adjust height if needed
                if top_x < 0:
                    top_x = 0
                if top_y < 0:
                    # push the top down and reduce height accordingly
                    overshoot = -top_y
                    top_y = 0
                    garment_h = max(1, garment_h - overshoot)
                if top_y + garment_h > frame_h:
                    garment_h = max(1, frame_h - top_y - 1)
                if top_x + garment_w > frame_w:
                    top_x = max(0, frame_w - garment_w)

                frame = overlay_transparent(frame, garment, top_x, top_y, garment_w, garment_h)

            cv2.imshow("Garment Overlay - Milestone 2 (placeholder garment)", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
