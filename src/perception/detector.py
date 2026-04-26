"""Orange-mass detector using a per-pixel 'orangeness' score in RGB."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Detection:
    found: bool
    x: int = 0
    y: int = 0
    area: int = 0
    score: float = 0.0  # mean orangeness of the detected blob


class OrangeDetector:
    """
    Orangeness score (per pixel, on float RGB in [0, 1]):
        s = R * (R - B) * (1 - |R - 2G|)         clipped to [0, 1]

    Intuition:
      - R high                  -> orange/red are strong in R
      - (R - B) high            -> excludes white/grey (where R==B)
      - (1 - |R - 2G|) high     -> orange has G ~ R/2; penalizes pure red
    """

    def __init__(self, min_area: int = 200, score_threshold: float = 0.15):
        self.min_area = min_area
        self.score_threshold = score_threshold

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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
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
