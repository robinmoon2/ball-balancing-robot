"""Camera abstraction: hides the source (webcam, Pi cam, video file)."""
from __future__ import annotations
import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    # Context manager support: `with Camera() as cam:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
