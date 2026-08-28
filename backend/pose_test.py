"""
Milestone 1 smoke test: confirm webcam + MediaPipe pose detection are working.

Run: python pose_test.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Skipping empty camera frame.")
                continue

            # MediaPipe expects RGB, OpenCV gives BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                # Quick rotation-bucket estimate for later multi-angle work:
                # compares shoulder landmark depth (z) to approximate facing direction
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                cv2.putText(
                    frame,
                    f"Shoulder width (rotation proxy): {shoulder_width:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Pose Detection - Milestone 1", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
