"""Abstract plate controller: ball state -> (roll, pitch) command.

Owns everything every control law needs the same way - target tracking,
the bad-dt / invalid-estimate safety policy, and the tilt-magnitude clamp.
Subclasses (ControllerPID, ControllerLQR) only implement `_compute`, the
part that actually differs: errors + velocities -> raw (roll, pitch).

Sign convention shared by every subclass (flip on hardware if needed):
  +pitch tilts the plate so the ball rolls toward +x
  +roll  tilts the plate so the ball rolls toward -y
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from src.utils import BallEstimate, PlateCommand

logger = logging.getLogger(__name__)


class Controller(ABC):
    def __init__(self, max_tilt_rad: float = math.radians(15)):
        self.max_tilt = max_tilt_rad
        self.target = (0.0, 0.0)  # mm
        self._last_command = PlateCommand(roll=0.0, pitch=0.0)

    def set_target(self, x: float, y: float) -> None:
        self.target = (x, y)

    def reset(self) -> None:
        """Clear any internal state (integrators...). Stateless by default."""

    def update(self, state: BallEstimate, dt: float) -> PlateCommand:
        if dt <= 0.0:
            # Same contract as Estimation/Actuation: never raise on a bad
            # tick, just hold the last command.
            logger.warning("non-positive dt (%.6f s); command ignored", dt)
            return self._last_command

        if not state.valid:
            # Safety: hold flat, and drop internal state so nothing winds up
            # while the ball is lost.
            self.reset()
            self._last_command = PlateCommand(roll=0.0, pitch=0.0)
            return self._last_command

        ex = state.x - self.target[0]
        ey = state.y - self.target[1]
        roll, pitch = self._compute(ex, ey, state.vx, state.vy, dt)

        # Hard clamp on the tilt MAGNITUDE: clamping each axis separately
        # lets a diagonal command reach sqrt(2) * max_tilt, which is where
        # the IK runs out of reach first.
        tilt = math.hypot(roll, pitch)
        if tilt > self.max_tilt:
            scale = self.max_tilt / tilt
            roll *= scale
            pitch *= scale

        self._last_command = PlateCommand(roll=roll, pitch=pitch)
        return self._last_command

    @abstractmethod
    def _compute(
        self, ex: float, ey: float, vx: float, vy: float, dt: float
    ) -> tuple[float, float]:
        """Position errors (mm) + velocities (mm/s) -> unclamped (roll, pitch) in rad."""
