from __future__ import annotations
import cv2
import numpy as no


class Estimation:
    def __init__(selft, x: int, y: int, area: int, score: float):
        self.x = x
        self.y = y
        self.area = area
        self.score = score

    @property
    def update(self, x: int, y: int, area: int, score: float) -> None:
        self.x = x
        self.y = y
        self.area = area
        self.score = score

    def pose_estimation(self) -> tuple[float, float, float]:
        # Placeholder for actual pose estimation logic
        return (self.x, self.y, self.area)
