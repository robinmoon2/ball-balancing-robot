"""LQR plate controller: full-state feedback u = -K x.

State x = [ex, ey, vx, vy] (mm, mm/s), input u = [roll, pitch] (rad).
K comes from the continuous-time Riccati equation on StateSpaceModel, whose
B matrix already encodes the base-class sign convention - so -K x yields
correctly-signed commands with no manual flipping.
"""

from __future__ import annotations

import math

import numpy as np

from src.control.controller import Controller
from src.control.LRQ_control.LQR import LQR
from src.utils import StateSpaceModel


class ControllerLQR(Controller):
    # Split of u = -K x into its position-feedback and velocity-feedback
    # parts (rad, before the tilt clamp): roll = roll_pos + roll_vel, etc.
    TERM_NAMES = ("roll_pos", "roll_vel", "pitch_pos", "pitch_vel")

    def __init__(
        self,
        model: StateSpaceModel,
        Q: np.ndarray,
        R: np.ndarray,
        max_tilt_rad: float = math.radians(15),
    ):
        super().__init__(max_tilt_rad)
        self.model = model
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.lqr = LQR(model, Q, R)
        self.K = self.lqr.get_gain()  # (2, 4)

    @classmethod
    def from_bryson(
        cls,
        model: StateSpaceModel,
        max_pos_mm: float,
        max_vel_mm_s: float,
        max_tilt_rad: float,
    ) -> "ControllerLQR":
        """Bryson's rule: weight each state/input by 1/(acceptable max)^2,
        so the tuning knobs are in physical units.

        Lower max_pos_mm  -> more aggressive centring.
        Lower max_vel_mm_s -> more damping.
        """
        Q = np.diag([1 / max_pos_mm**2] * 2 + [1 / max_vel_mm_s**2] * 2)
        R = np.diag([1 / max_tilt_rad**2] * 2)
        return cls(model, Q, R, max_tilt_rad)

    def coefficients(self) -> dict:
        c = super().coefficients()
        c.update({
            "g_eff": self.model.g_eff,
            "Q_diag": np.diag(self.Q).tolist(),
            "R_diag": np.diag(self.R).tolist(),
            "K": self.K.tolist(),  # rows: roll, pitch; cols: x, y, vx, vy
        })
        return c

    def _compute(self, ex, ey, vx, vy, dt):
        u_pos = -self.K[:, :2] @ np.array([ex, ey])
        u_vel = -self.K[:, 2:] @ np.array([vx, vy])
        self.last_terms = {
            "roll_pos": float(u_pos[0]), "roll_vel": float(u_vel[0]),
            "pitch_pos": float(u_pos[1]), "pitch_vel": float(u_vel[1]),
        }
        u = u_pos + u_vel
        return float(u[0]), float(u[1])
