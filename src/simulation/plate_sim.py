"""2D physics: ball rolling on a plate with small tilt angles."""

from __future__ import annotations
import math

from src.utils import SimParams, SimState


class PlateSim:
    """Point-mass ball on a tilted disk. Small-angle approximation.

    State: position (x, y) in mm, velocity (vx, vy) in mm/s.
    Input: commanded (roll, pitch) in radians.
    """

    def __init__(self, params: SimParams | None = None, seed: int | None = 0):
        import random

        self.p = params or SimParams()
        self.s = SimState()
        self._rng = random.Random(seed)
        self._cmd_roll = 0.0
        self._cmd_pitch = 0.0

    def reset(self, x: float = 30.0, y: float = -20.0) -> None:
        self.s = SimState(x=x, y=y)
        self._cmd_roll = 0.0
        self._cmd_pitch = 0.0

    def set_command(self, roll: float, pitch: float) -> None:
        self._cmd_roll = roll
        self._cmd_pitch = pitch

    def step(self, dt: float) -> None:
        if not self.s.on_plate:
            return

        # First-order actuator lag: actual angle chases commanded
        tau = max(self.p.actuator_lag_tau, 1e-6)
        alpha = dt / (tau + dt)
        self.s.roll += alpha * (self._cmd_roll - self.s.roll)
        self.s.pitch += alpha * (self._cmd_pitch - self.s.pitch)

        # Accelerations (mm/s^2). Sign matches PlateController convention:
        # +pitch -> ball accelerates in +x direction.
        k = self.p.rolling_factor * self.p.g
        ax = k * math.sin(self.s.pitch) - self.p.friction * self.s.vx
        ay = -k * math.sin(self.s.roll) - self.p.friction * self.s.vy

        # Semi-implicit Euler (more stable than explicit Euler)
        self.s.vx += ax * dt
        self.s.vy += ay * dt
        self.s.x += self.s.vx * dt
        self.s.y += self.s.vy * dt

        # Off-plate check
        if math.hypot(self.s.x, self.s.y) > self.p.plate_radius:
            self.s.on_plate = False

    def measure(self) -> tuple[float, float, bool]:
        """Returns (x_meas, y_meas, found) — what perception would output."""
        if not self.s.on_plate:
            return (0.0, 0.0, False)
        if self._rng.random() < self.p.detection_dropout_prob:
            return (0.0, 0.0, False)
        n = self.p.measurement_noise_std
        return (
            self.s.x + self._rng.gauss(0, n),
            self.s.y + self._rng.gauss(0, n),
            True,
        )
