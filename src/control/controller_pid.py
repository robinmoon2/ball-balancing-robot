"""PID plate controller: two independent PIDs, X-error -> pitch, Y-error -> roll."""

from __future__ import annotations

import math

from src.control.controller import Controller
from src.utils import PID


class ControllerPID(Controller):
    """With the base-class sign convention (+pitch rolls the ball toward +x),
    driving x toward 0 when x>0 needs NEGATIVE pitch => pitch = -PID(x).
    +roll rolls the ball toward -y, so roll = +PID(y)."""

    # Raw PID contributions, BEFORE the pitch sign flip.
    TERM_NAMES = ("x_p", "x_i", "x_d", "y_p", "y_i", "y_d")

    def __init__(self, pid_x: PID, pid_y: PID, max_tilt_rad: float = math.radians(15)):
        super().__init__(max_tilt_rad)
        self.pid_x = pid_x
        self.pid_y = pid_y

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()

    def coefficients(self) -> dict:
        c = super().coefficients()
        for axis, pid in (("x", self.pid_x), ("y", self.pid_y)):
            c.update({f"{axis}_kp": pid.kp, f"{axis}_ki": pid.ki, f"{axis}_kd": pid.kd,
                      f"{axis}_integral_limits": pid.integral_limits,
                      f"{axis}_output_limits": pid.output_limits})
        return c

    def _compute(self, ex, ey, vx, vy, dt):
        pitch = -self.pid_x.update(ex, vx, dt)
        roll = self.pid_y.update(ey, vy, dt)
        self.last_terms = {
            "x_p": self.pid_x.last_p, "x_i": self.pid_x.last_i, "x_d": self.pid_x.last_d,
            "y_p": self.pid_y.last_p, "y_i": self.pid_y.last_i, "y_d": self.pid_y.last_d,
        }
        return roll, pitch
