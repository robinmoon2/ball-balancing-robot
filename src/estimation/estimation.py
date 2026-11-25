"""Ball state estimator: noisy, intermittent detections -> clean (pos, vel).

Implements the contract in estimation.md. This class owns the *policy*
(when to initialize, when to predict-only, when the estimate is trustworthy);
KalmanFilter owns the *math*.

Frame and units are the plate frame in mm, as produced upstream - this class
does no pixel conversion (estimation.md section 2).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.estimation.kalman_filter import KalmanFilter
from src.utils import BallEstimate

logger = logging.getLogger(__name__)


class Estimation:
    """Constant-velocity estimator over intermittent (x, y) measurements."""

    def __init__(
        self,
        process_noise_std: float = 1500.0,
        measurement_noise_std: float = 1.5,
        warmup_ticks: int = 5,
        max_dropout_seconds: float = 0.25,
        initial_velocity_std: float = 300.0,
    ):
        # process_noise_std (mm/s^2): the acceleration the constant-velocity
        # model does NOT know about, i.e. plate tilt. A ball rolling on a
        # plate at the 15 deg tilt limit sees a*g*sin(tilt) ~ 1800 mm/s^2
        # (a = 5/7 for a rolling sphere), so this is the right order.
        self.process_noise_std = process_noise_std
        # measurement_noise_std (mm): detector centroid scatter on a still ball.
        self.measurement_noise_std = measurement_noise_std
        self.warmup_ticks = warmup_ticks
        self.max_dropout_seconds = max_dropout_seconds
        self.initial_velocity_std = initial_velocity_std

        self.reset()

    def reset(self) -> None:
        """Drop all state. The next measurement re-initializes the filter."""
        self._kf: KalmanFilter | None = None
        self._t: float = 0.0  # timestamp the filter state is propagated to
        self._t_last_measurement: float = 0.0
        self._ticks: int = 0  # measurements accepted since reset
        self._misses: int = 0  # consecutive frames without a measurement
        self._last = BallEstimate(0.0, 0.0, 0.0, 0.0, 0.0, valid=False)

    @property
    def consecutive_misses(self) -> int:
        return self._misses

    def update(
        self, measurement: tuple[float, float] | None, t: float) -> BallEstimate:
        """Advance the estimate to time `t` and return it."""
        meas = self._sanitize(measurement)

        # First call ever (or first after a reset): nothing to propagate.
        if self._kf is None:
            if meas is None:
                self._misses += 1
                self._last = BallEstimate(0.0, 0.0, 0.0, 0.0, t, valid=False)
                return self._last
            self._initialize(meas, t)
            return self._emit(t)

        dt = t - self._t
        if dt <= 0.0:
            logger.warning(
                "non-monotonic timestamp (t=%.6f, last=%.6f, dt=%.6f); tick ignored",
                t,
                self._t,
                dt,
            )
            return self._last

        self._kf.predict(dt)
        self._t = t

        if meas is None:
            self._misses += 1
        else:
            self._kf.update(meas[0], meas[1])
            self._t_last_measurement = t
            self._ticks += 1
            self._misses = 0

        return self._emit(t)

    def _initialize(self, meas: tuple[float, float], t: float) -> None:
        self._kf = KalmanFilter(
            initial_x=meas[0],
            initial_y=meas[1],
            accel_variance=self.process_noise_std**2,
            R=np.eye(2) * self.measurement_noise_std**2,
        )
        # KalmanFilter's default P is tuned for pixels; restate it in mm.
        # Position is known to measurement precision, velocity is unknown.
        self._kf.P = np.diag(
            [
                self.measurement_noise_std**2,
                self.measurement_noise_std**2,
                self.initial_velocity_std**2,
                self.initial_velocity_std**2,
            ]
        ).astype(float)
        self._t = t
        self._t_last_measurement = t
        self._ticks = 1
        self._misses = 0

    def _emit(self, t: float) -> BallEstimate:
        assert self._kf is not None
        x, y, vx, vy = (float(v) for v in self._kf.get_current_state().flatten())
        dropout = t - self._t_last_measurement
        valid = self._ticks >= self.warmup_ticks and dropout <= self.max_dropout_seconds
        self._last = BallEstimate(
            x=x, y=y, vx=vx, vy=vy, t=t, valid=valid, cov=self._kf.P.copy()
        )
        return self._last

    @staticmethod
    def _sanitize(
        measurement: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """A NaN/inf measurement is a dropout, not a position."""
        if measurement is None:
            return None
        x, y = float(measurement[0]), float(measurement[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            logger.warning("non-finite measurement (%r, %r); treated as dropout", x, y)
            return None
        return (x, y)
