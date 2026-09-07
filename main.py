"""Live demo: show camera feed with a green dot on the most orange mass."""

from __future__ import annotations
import cv2
import numpy as np
import os
import time
from src.perception.camera import Camera
from src.perception.detector import OrangeDetector

from src.estimation.kalman_filter import KalmanFilter

# global variables for Kalman Filter
g = 0.0
MEASUREMENT_NOISE_VARIANCE = 4.0
R = np.eye(2) * MEASUREMENT_NOISE_VARIANCE

KF = None


def draw_overlay(frame, detection, dt) -> None:
    global KF
    if detection.found:
        if KF is None:
            KF = KalmanFilter(
                initial_x=detection.x,
                initial_y=detection.y,
                accel_variance=250000.0,
                g=9.81,
                R=R,
            )  # Initialize Kalman Filter with gravit
            return  # Skip drawing for the first frame to initialize the filter
        KF.predict(dt=dt)  # Predict the next state
        predicted_state = KF.get_current_state()
        print(
            f"pos=({predicted_state[0]},{predicted_state[1]}) vel=({predicted_state[2]},{predicted_state[3]}) dt={dt * 1000:}ms"
        )
        predicted_state = (
            predicted_state.flatten()
        )  # Flatten the state for easier access
        cv2.circle(
            frame,
            (int(predicted_state[0]), int(predicted_state[1])),
            8,
            (255, 0, 0),
            -1,
        )  # Draw predicted position

        KF.update(detection.x, detection.y)  # Update with the new measurement
        update_state = KF.get_current_state()
        update_state = update_state.flatten()  # Flatten the state for easier access
        print(
            f"pos=({update_state[0]},{update_state[1]}) vel=({update_state[2]},{update_state[3]}) dt={dt * 1000:.0f}ms"
        )

        cv2.circle(
            frame, (int(update_state[0]), int(update_state[1])), 8, (0, 0, 255), -1
        )  # Draw updated position
        cv2.circle(frame, (detection.x, detection.y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (detection.x, detection.y), 12, (0, 255, 0), 2)
        label = f"({detection.x},{detection.y}) area={detection.area} s={detection.score:.2f}"
        cv2.putText(
            frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    else:
        if KF is not None:
            KF.predict(dt=dt)  # Predict the next state even if no detection
            predicted_state = KF.get_current_state()
            predicted_state = (
                predicted_state.flatten()
            )  # Flatten the state for easier access
            cv2.circle(
                frame,
                (int(predicted_state[0]), int(predicted_state[1])),
                8,
                (255, 0, 0),
                -1,
            )  # Draw predicted position
        cv2.putText(
            frame,
            "No orange detected",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def main(source: int | str = 0, output_path: str = "output.avi") -> None:
    last_time = time.time()
    detector = OrangeDetector(min_area=200, score_threshold=0.15)
    display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    with Camera(source) as cam:
        while True:
            frame = cam.read()
            now = time.time()
            dt = now - last_time
            last_time = now
            if frame is None:
                print("Frame grab failed.")
                break
            print("frame size:", frame.shape)
            detection = detector.detect(frame)
            draw_overlay(frame, detection, dt)

            if display:
                cv2.imshow("Perception - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Start of the script ")
    main(0)
