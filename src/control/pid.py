"""Pure PID controller. No I/O, no time.time(). dt is passed explicitly."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limits: tuple[float, float] = (-float("inf"), float("inf"))
    integral_limits: tuple[float, float] = (-float("inf"), float("inf"))
    derivative_filter: float = 0.0  # 0 = no filter, 1 = full filter (no derivative)

    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)
    _prev_derivative: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._prev_derivative = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")

        # Integral with clamping (anti-windup)
        self._integral += error * dt
        lo, hi = self.integral_limits
        self._integral = max(lo, min(hi, self._integral))

        # Derivative on error, with optional low-pass smoothing
        if self._prev_error is None:
            raw_deriv = 0.0
        else:
            raw_deriv = (error - self._prev_error) / dt

        a = self.derivative_filter
        derivative = a * self._prev_derivative + (1 - a) * raw_deriv

        # Output
        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Output saturation
        lo, hi = self.output_limits
        output = max(lo, min(hi, output))

        # Save state
        self._prev_error = error
        self._prev_derivative = derivative

        return output
