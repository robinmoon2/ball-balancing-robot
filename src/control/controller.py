"""Plate controller: ball state -> (roll, pitch) commands."""

from __future__ import annotations
from dataclasses import dataclass
import math

from .pid import PID


@dataclass
class BallState:
    x: float  # mm, plate frame
    y: float  # mm
    vx: float = 0.0  # mm/s
    vy: float = 0.0
    found: bool = True


@dataclass
class PlateCommand:
    roll: float  # rad, rotation about plate X axis
    pitch: float  # rad, rotation about plate Y axis


class PlateController:
    """Two independent PIDs: X-error -> pitch, Y-error -> roll.

    Sign convention (you may need to flip on hardware):
      +pitch tilts plate so ball rolls toward +x  ⇒ to drive x toward 0,
        when x>0 we need NEGATIVE pitch ⇒ pitch = -PID(x).
      Same logic for roll on Y.
    """

    def __init__(
        self,
        pid_x: PID,
        pid_y: PID,
        max_tilt_rad: float = math.radians(15),
    ):
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.max_tilt = max_tilt_rad
        self.target = (0.0, 0.0)  # mm

    def set_target(self, x: float, y: float) -> None:
        self.target = (x, y)

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()

    def update(self, state: BallState, dt: float) -> PlateCommand:
        if not state.found:
            # Safety: hold flat. Reset integrators to avoid windup during loss.
            self.reset()
            return PlateCommand(roll=0.0, pitch=0.0)

        ex = state.x - self.target[0]
        ey = state.y - self.target[1]

        pitch = -self.pid_x.update(ex, dt)
        roll = self.pid_y.update(ey, dt)

        # Hard clamp for safety
        pitch = max(-self.max_tilt, min(self.max_tilt, pitch))
        roll = max(-self.max_tilt, min(self.max_tilt, roll))

        return PlateCommand(roll=roll, pitch=pitch)
