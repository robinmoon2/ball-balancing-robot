"""Live demo: show camera feed with a green dot on the most orange mass."""

from __future__ import annotations
import cv2

from camera import Camera
from detector import OrangeDetector


def draw_overlay(frame, detection) -> None:
    if detection.found:
        cv2.circle(frame, (detection.x, detection.y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (detection.x, detection.y), 12, (0, 255, 0), 2)
        label = f"({detection.x},{detection.y}) area={detection.area} s={detection.score:.2f}"
        cv2.putText(
            frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame,
            "No orange detected",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def main(source: int | str = 0) -> None:
    detector = OrangeDetector(min_area=200, score_threshold=0.15)
    with Camera(source) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                print("Frame grab failed.")
                break

            detection = detector.detect(frame)
            draw_overlay(frame, detection)

            cv2.imshow("Perception - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)
