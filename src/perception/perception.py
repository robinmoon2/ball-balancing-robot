"""Perception layer: camera frame -> ball position in plate-frame mm.

Wraps Camera (frame capture) and OrangeDetector (frame -> pixel blob) behind
one call per loop tick, and applies Calibration so the output is exactly
what Estimation.update() wants as a measurement - no separate pixel->mm
conversion step anywhere else in the pipeline.
"""

from __future__ import annotations

from src.perception.camera import Camera
from src.perception.detector import OrangeDetector
from src.utils import Calibration, Detection


class Perception:
    def __init__(
        self,
        calibration: Calibration,
        camera: Camera | None = None,
        detector: OrangeDetector | None = None,
    ):
        self.camera = camera or Camera()
        self.detector = detector or OrangeDetector()
        self.calibration = calibration
        self._last_detection = Detection(found=False)

    def read(self) -> tuple[float, float] | None:
        """Grab one frame and return the ball's (x, y) in plate-frame mm,
        or None if the frame failed to grab or no ball was found."""
        frame = self.camera.read()
        if frame is None:
            self._last_detection = Detection(found=False)
            return None

        self._last_detection = self.detector.detect(frame)
        if not self._last_detection.found:
            return None

        return self.calibration.to_mm(self._last_detection.x, self._last_detection.y)

    @property
    def last_detection(self) -> Detection:
        """Raw pixel-space result of the last read() call, for debug overlays."""
        return self._last_detection

    def release(self) -> None:
        self.camera.release()

    def __enter__(self) -> "Perception":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
