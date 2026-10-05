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
        use_yuv: bool = False,
    ):
        self.camera = camera or Camera()
        self.detector = detector or OrangeDetector()
        self.calibration = calibration
        # use_yuv picks the cheap chroma-plane detector over the BGR one.
        # Off by default because its thresholds are rig-specific - see
        # OrangeDetector.detect_yuv.
        self.use_yuv = use_yuv
        self._last_detection = Detection(found=False)
        self._last_capture_t = 0.0

    def read(self) -> tuple[float, float] | None:
        """Grab the newest frame and return the ball's (x, y) in plate-frame
        mm, or None if the frame failed to grab or no ball was found.

        The frame's capture time is recorded in `last_capture_t`. Feed THAT
        to the estimator, not time.monotonic() after the call: the frame
        describes where the ball was when it was taken, and fusing it under
        a later timestamp injects a position error proportional to the
        ball's speed times the loop latency.
        """
        frame = self.camera.read_frame()
        if frame is None:
            self._last_detection = Detection(found=False)
            return None

        self._last_capture_t = frame.t_capture
        self._last_detection = (
            self.detector.detect_yuv(frame) if self.use_yuv
            else self.detector.detect(frame.bgr)
        )
        if not self._last_detection.found:
            return None

        return self.calibration.to_mm(self._last_detection.x, self._last_detection.y)

    @property
    def last_capture_t(self) -> float:
        """time.monotonic() at which the last frame read() saw was captured."""
        return self._last_capture_t

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
