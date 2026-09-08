"""Actuation layer: a PlateCommand -> per-arm servo angles -> PWM.

This is the bottom of the stack and the only place that knows the machine is
a 3-arm parallel platform. It takes the roll/pitch the controller wants and
is responsible for everything below that: inverse kinematics, rejecting
unreachable poses, slew-limiting, and writing to the servos.

Angles here are DIFFERENTIAL, not absolute. At construction the flat pose is
solved once and kept as `_neutral_solution`; every later command is issued as

    commanded = arm's offset_rad + (IK(command) - IK(flat))

so each arm's calibrated offset_rad *is* the zero, and IK is only ever asked
for the *change* from flat. Any systematic error in the geometry (h, L, L1,
L2, L3) cancels in that subtraction - it can no longer push the plate away
from neutral, it only scales how much tilt a given command produces, which
the PID gains absorb.

All per-arm quantities are np.ndarray of shape (n_arms,) in that frame.
Arm.set_angle subtracts offset_rad again right before writing, so what the
servo finally receives is the pure delta (0 = that servo's calibrated flat).
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
    max_command_rad:
        Hard safety ceiling: no arm is ever commanded more than
        max_command_rad away from its OWN calibrated neutral (its
        servo's offset_rad), regardless of what IK computes or what that
        servo's full mechanical range would otherwise allow.
    dry_run:
        Solve and guard as normal, but never write to the PWM lines.
    """

    def __init__(
        self,
        arms: np.ndarray,
        plate_radius: float,
        neutral_height: float,
        max_rate_rad_s: float | None = 6.0,
        max_command_rad: float = np.pi,
        dry_run: bool = False,
    ):
        self.arms = np.asarray(arms, dtype=object)
        if self.arms.ndim != 1 or self.arms.size == 0:
            raise ValueError("arms must be a non-empty 1-D array of Arm objects")

        self.plate_radius = float(plate_radius)
        self.neutral_height = float(neutral_height)
        self.max_rate_rad_s = max_rate_rad_s
        self.max_command_rad = float(max_command_rad)
        self.dry_run = dry_run

        n = self.arms.size

        # Each arm's calibrated flat position. This is the zero of the
        # differential frame, and where the arms start.
        self._offsets = np.fromiter(
            (a.servo.offset_rad for a in self.arms), dtype=float, count=n
        )
        self._range_min = np.maximum(0.0, self._offsets - self.max_command_rad)
        self._range_max = np.minimum(np.pi, self._offsets + self.max_command_rad)

        # IK's absolute answer for the flat pose at neutral_height. Every
        # later solve is expressed as a delta from this, so its absolute
        # value never reaches a servo - only differences from it do.
        self._neutral_solution = self._solve_absolute(
            PlateCommand(roll=0.0, pitch=0.0), self.neutral_height
        )
        if not np.all(np.isfinite(self._neutral_solution)):
            raise ValueError(
                "the flat pose at neutral_height="
                f"{self.neutral_height} is unreachable for this geometry "
                "(plate_radius/L1/L2/L3); the differential reference frame "
                "cannot be established"
            )

        self._angles = self._offsets.copy()
        self._saturated = np.zeros(n, dtype=bool)
        self._reachable = True

        self._write(self._angles)

    @property
    def n_arms(self) -> int:
        return int(self.arms.size)

    @property
    def last_angles(self) -> np.ndarray:
        """Angles last written to the servos (rad, raw geometric frame -
        see the module docstring; NOT each servo's own centered frame)."""
        return self._angles.copy()

    @property
    def last_deltas(self) -> np.ndarray:
        """What the arms actually did, as each arm's movement away from its
        own calibrated flat position (rad). 0 = that arm is level. This is
        the number to log - `last_angles` is offset by each servo's own
        calibration and so isn't comparable between arms."""
        return self._angles - self._offsets

    @property
    def saturated(self) -> np.ndarray:
        """Per-arm bool: did the last command hit that servo's travel limit?"""
        return self._saturated.copy()

    @property
    def reachable(self) -> bool:
        """Was the last commanded pose inside the arms' workspace?"""
        return self._reachable

    def _solve_absolute(self, command: PlateCommand, height: float | None = None) -> np.ndarray:
        """Raw IK, in its own absolute frame. Only used to build and to
        difference against `_neutral_solution` - never commanded directly."""
        z = self.neutral_height if height is None else float(height)
        with np.errstate(invalid="ignore"):
            angles = incline_board(
                pitch=command.pitch,
                roll=command.roll,
                height=z,
                arms=self.arms,
                L=self.plate_radius,
            )
        return np.asarray(angles, dtype=float)

    def solve(self, command: PlateCommand, height: float | None = None) -> np.ndarray:
        """PlateCommand -> commanded angles, as offset + delta-from-flat.
        Pure: touches no hardware.

        A flat command at neutral_height returns exactly each arm's
        offset_rad, i.e. zero movement from the startup pose. May contain
        NaN for an arm outside the workspace - `apply` rejects those before
        they reach a servo.
        """
        return self._offsets + (
            self._solve_absolute(command, height) - self._neutral_solution
        )

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
        logger.info(
                f"target pose : {target}",
            )
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
            print(
                f"arm {arm.servo.name}: commanded {angle:+.4f} rad "
                f"(delta from flat: {angle - arm.servo.offset_rad:+.4f} rad)"
            )

    def __enter__(self) -> "Actuation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()