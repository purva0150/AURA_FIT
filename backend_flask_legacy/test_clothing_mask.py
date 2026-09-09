import cv2
import numpy as np

from tryon.pose import PoseEstimator
from tryon import compose


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    with PoseEstimator(segmentation=True) as pose:
        while True:
            ok, frame = cap.read()

            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            result = pose.process(frame)

            if result is not None:
                h, w = frame.shape[:2]

                mask = compose.torso_clothing_mask(
                    result,
                    (w, h),
                )

                # Convert mask to visible grayscale image.
                mask_img = np.clip(mask * 255, 0, 255).astype(np.uint8)

                # Show the mask as white over black.
                mask_view = cv2.cvtColor(
                    mask_img,
                    cv2.COLOR_GRAY2BGR,
                )

                # Also show the mask over the camera frame.
                overlay = frame.copy()

                red = np.zeros_like(frame)
                red[:, :, 2] = 255

                alpha = (mask * 0.45)[:, :, None]

                overlay = (
                    frame * (1.0 - alpha)
                    + red * alpha
                ).astype(np.uint8)

                cv2.putText(
                    overlay,
                    "TORSO CLOTHING MASK",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Camera + Clothing Mask", overlay)
                cv2.imshow("Clothing Mask", mask_view)

            else:
                cv2.imshow("Camera + Clothing Mask", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()