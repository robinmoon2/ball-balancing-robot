"""PID plate controller: two independent PIDs, X-error -> pitch, Y-error -> roll."""

from __future__ import annotations

import math

from src.control.controller import Controller
from src.utils import PID


class ControllerPID(Controller):
    """With the base-class sign convention (+pitch rolls the ball toward +x),
    driving x toward 0 when x>0 needs NEGATIVE pitch => pitch = -PID(x).
    +roll rolls the ball toward -y, so roll = +PID(y)."""

    def __init__(self, pid_x: PID, pid_y: PID, max_tilt_rad: float = math.radians(15)):
        super().__init__(max_tilt_rad)
        self.pid_x = pid_x
        self.pid_y = pid_y

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()

    def _compute(self, ex, ey, vx, vy, dt):
        pitch = -self.pid_x.update(ex, vx, dt)
        roll = self.pid_y.update(ey, vy, dt)
        return roll, pitch
