"""Orange-mass detector using a per-pixel 'orangeness' score in RGB."""

from __future__ import annotations
import cv2
import numpy as np

from src.utils import Detection


class OrangeDetector:
    """
    Orangeness score (per pixel, on float RGB in [0, 1]):
        s = R * (R - B) * (1 - |R - 2G|)         clipped to [0, 1]

    Intuition:
      - R high                  -> orange/red are strong in R
      - (R - B) high            -> excludes white/grey (where R==B)
      - (1 - |R - 2G|) high     -> orange has G ~ R/2; penalizes pure red
    """

    def __init__(
        self,
        min_area: int = 200,
        score_threshold: float = 0.15,
        u_max: int | None = None,
        v_min: int | None = None,
        y_min: int = 40,
    ):
        self.min_area = min_area
        self.score_threshold = score_threshold
        # Chroma bounds for detect_yuv. None = not calibrated yet; run
        # `python -m src.perception.calibrate_yuv` with the ball on the
        # plate to measure them under your own lighting.
        self.u_max = u_max
        self.v_min = v_min
        self.y_min = y_min

    @staticmethod
    def _orangeness(frame_bgr: np.ndarray) -> np.ndarray:
        f = frame_bgr.astype(np.float32) / 255.0
        B, G, R = f[..., 0], f[..., 1], f[..., 2]
        s = R * np.clip(R - B, 0, 1) * np.clip(1.0 - np.abs(R - 2.0 * G), 0, 1)
        return np.clip(s, 0.0, 1.0)

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        score_map = self._orangeness(frame_bgr)
        mask = (score_map > self.score_threshold).astype(np.uint8) * 255

        # Clean noise: open (erode then dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(found=False)

        largest = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(largest))
        if area < self.min_area:
            return Detection(found=False)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return Detection(found=False)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Mean score inside the blob (a confidence proxy)
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mean_score = float(score_map[blob_mask == 255].mean())

        return Detection(found=True, x=cx, y=cy, area=area, score=mean_score)

    def detect_yuv(self, frame) -> Detection:
        """Same job as detect(), straight off the I420 chroma planes.

        ~100x cheaper than detect(): no JPEG decode, no float32 temporaries,
        and U/V are already half-resolution, so this thresholds 320x240
        instead of scanning 640x480x3. Takes a camera.Frame, not an array.

        The DISCRIMINANT IS NOT THE SAME as detect(). detect() separates
        orange from pure red via the (1 - |R - 2G|) term; a high-V threshold
        alone does not - red and orange both sit high in V. The y_min floor
        rejects dark red, but if anything red shares the plate you want
        detect() or a tighter u_max. Bounds must be calibrated for your ball
        and lighting: see calibrate_yuv.

        Returns pixel coordinates in FULL-resolution units, so it is
        interchangeable with detect() as far as Calibration is concerned.
        """
        if self.u_max is None or self.v_min is None:
            raise ValueError(
                "detect_yuv needs calibrated chroma bounds; u_max/v_min are unset. "
                "Run `python -m src.perception.calibrate_yuv` first, or use detect()."
            )

        mask = cv2.inRange(frame.v, self.v_min, 255)
        cv2.bitwise_and(mask, cv2.inRange(frame.u, 0, self.u_max), dst=mask)
        # Luma is full-res; sample it at chroma resolution to reject dark
        # pixels that happen to land in the orange chroma box.
        cv2.bitwise_and(mask, cv2.inRange(frame.y[::2, ::2], self.y_min, 255), dst=mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(found=False)

        largest = max(contours, key=cv2.contourArea)
        # Chroma is half-resolution in each axis, so one chroma pixel is
        # four full-res pixels: scale the area before comparing to min_area.
        area = int(cv2.contourArea(largest)) * 4
        if area < self.min_area:
            return Detection(found=False)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return Detection(found=False)
        cx = int(2 * M["m10"] / M["m00"])
        cy = int(2 * M["m01"] / M["m00"])

        return Detection(found=True, x=cx, y=cy, area=area, score=1.0)
