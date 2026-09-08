"""Actuation layer: a PlateCommand -> per-arm servo angles -> PWM.

This is the bottom of the stack and the only place that knows the machine is
a 3-arm parallel platform. It takes the roll/pitch the controller wants and
is responsible for everything below that: inverse kinematics, rejecting
unreachable poses, slew-limiting, and writing to the servos.

All per-arm quantities are np.ndarray of shape (n_arms,), angles in radians
in each servo's centred frame (0 = neutral / plate flat), matching
Servo.set_angle.
"""

from __future__ import annotations

import logging

import numpy as np

from src.control.arm import Arm
from src.control.controller import PlateCommand
from src.control.inverse_kinematics import incline_board

logger = logging.getLogger(__name__)

class Actuation:
    """Drives the platform's arms to hold a commanded plate pose.

    Parameters
    ----------
    arms:
        The Arm objects, as a 1-D np.ndarray.
    plate_radius:
        L, plate centre -> spherical joint (mm).
    neutral_height:
        Default plate-centre height (mm) 
    max_rate_rad_s:
        Per-joint slew limit. None disables it.
    dry_run:
        Solve and guard as normal, but never write to the PWM lines.
    """

    def __init__(
        self,
        arms: np.ndarray,
        plate_radius: float,
        neutral_height: float,
        max_rate_rad_s: float | None = 6.0,
        dry_run: bool = False,
    ):
        self.arms = np.asarray(arms, dtype=object)
        if self.arms.ndim != 1 or self.arms.size == 0:
            raise ValueError("arms must be a non-empty 1-D array of Arm objects")

        self.plate_radius = float(plate_radius)
        self.neutral_height = float(neutral_height)
        self.max_rate_rad_s = max_rate_rad_s
        self.dry_run = dry_run

        n = self.arms.size
        self._range_min = np.fromiter(
            (a.servo.range_min for a in self.arms), dtype=float, count=n
        )
        self._range_max = np.fromiter(
            (a.servo.range_max for a in self.arms), dtype=float, count=n
        )
        # Seed from the servos' real current angle, not an assumed neutral -
        # matters if this object gets constructed mid-flight.
        self._angles = np.fromiter(
            (a.servo.get_angle() for a in self.arms), dtype=float, count=n
        )
        self._saturated = np.zeros(n, dtype=bool)
        self._reachable = True

    @property
    def n_arms(self) -> int:
        return int(self.arms.size)

    @property
    def last_angles(self) -> np.ndarray:
        """Angles last written to the servos (rad, centred frame)."""
        return self._angles.copy()

    @property
    def saturated(self) -> np.ndarray:
        """Per-arm bool: did the last command hit that servo's travel limit?"""
        return self._saturated.copy()

    @property
    def reachable(self) -> bool:
        """Was the last commanded pose inside the arms' workspace?"""
        return self._reachable

    def solve(self, command: PlateCommand, height: float | None = None) -> np.ndarray:
        """PlateCommand -> raw IK joint angles. Pure: touches no hardware.

        May contain NaN for an arm outside the workspace - `apply` is what
        rejects those before they reach a servo.
        """
        z = self.neutral_height if height is None else float(height)
        with np.errstate(invalid="ignore"):
            angles = incline_board(pitch=command.pitch, roll=command.roll, height=z, arms=self.arms)
        return np.asarray(angles, dtype=float)

    def apply(self, command: PlateCommand, dt: float, height: float | None = None) -> np.ndarray:
        """Solve, guard, and write. Returns the angles actually commanded.

        Guards, in order: dt must be positive; the pose must be reachable
        (no NaN from `solve`); the per-tick step is capped to what the
        servos can physically track; the result is clamped to each servo's
        travel. 
        A rejected command leaves the arms exactly where they are.
        """
        if dt <= 0.0:
            logger.warning("non-positive dt (%.6f s); command ignored", dt)
            return self.last_angles

        target = self.solve(command, height)

        self._reachable = bool(np.all(np.isfinite(target)))
        if not self._reachable:
            logger.warning(
                "unreachable pose (roll=%.3f, pitch=%.3f rad); holding last angles",
                command.roll, command.pitch,
            )
            return self.last_angles

        if self.max_rate_rad_s is not None:
            max_step = self.max_rate_rad_s * dt
            target = self._angles + np.clip(target - self._angles, -max_step, max_step)

        clamped = np.clip(target, self._range_min, self._range_max)
        self._saturated = ~np.isclose(clamped, target)
        if np.any(self._saturated):
            logger.warning("servo travel limit hit on arm(s) %s", np.flatnonzero(self._saturated))

        self._write(clamped)
        return self.last_angles

    def hold_flat(self, dt: float, height: float | None = None) -> np.ndarray:
        """Level the plate - the safe fallback when the ball estimate is invalid."""
        return self.apply(PlateCommand(roll=0.0, pitch=0.0), dt, height)

    def release(self) -> None:
        """Cut PWM on every servo so the arms go limp on shutdown."""
        if self.dry_run:
            return
        for arm in self.arms:
            arm.servo.release()

    def _write(self, angles: np.ndarray) -> None:
        self._angles = angles
        if self.dry_run:
            return
        for arm, angle in zip(self.arms, angles):
            arm.set_angle(float(angle))

    def __enter__(self) -> "Actuation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()