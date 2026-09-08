"""Shared dataclasses used across perception -> estimation -> control ->
actuation and the simulator.

Kept in one place so every layer imports the same definition instead of
each owning its own copy. Grouped by the file each one used to live in;
each origin file now does `from utils import <Name>` (or `from src.utils`,
matching that file's own import style) instead of defining it locally.
"""


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- from control/pid.py ----------------------------------------------------


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


# --- from control/controller.py ---------------------------------------------


@dataclass
class PlateCommand:
    roll: float  # rad, rotation about plate X axis
    pitch: float  # rad, rotation about plate Y axis


# --- from control/inverse_kinematics.py -------------------------------------


@dataclass
class PlateOrientation:
    n: tuple[float, float, float]  # (alpha, beta, gamma), need not be unit
    h: float  # height of plate center above the motor-axis (z=0) plane


@dataclass
class VectorPosition:
    x: float
    y: float
    z: float


# --- from perception/detector.py --------------------------------------------


@dataclass
class Detection:
    found: bool
    x: int = 0
    y: int = 0
    area: int = 0
    score: float = 0.0  # mean orangeness of the detected blob


# --- from perception/perception.py ------------------------------------------


@dataclass
class Calibration:
    """Pixel -> plate-frame mm, for a camera on-axis with the plate normal
    (looking straight down from above, or straight up from below, as on
    this rig). Perspective correction (a homography) isn't needed at this
    angle - only scale, an origin, and possibly a mirrored axis.

    Measure by placing the ball at a known plate-frame position (e.g. the
    center, and one point at a known radius) and reading off its pixel
    coords from Detection.
    """

    origin_px: tuple[float, float]  # pixel coords of the plate center (0, 0)
    mm_per_px: float  # physical size of one pixel, at the plate
    flip_x: bool = False
    flip_y: bool = True  # image rows grow downward; plate +Y is up

    def to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        dx = x_px - self.origin_px[0]
        dy = y_px - self.origin_px[1]
        x_mm = (-dx if self.flip_x else dx) * self.mm_per_px
        y_mm = (-dy if self.flip_y else dy) * self.mm_per_px
        return x_mm, y_mm


# --- from estimation/estimation.py ------------------------------------------


@dataclass
class BallEstimate:
    """Estimator output. `valid` is a quality flag, not a safety decision -
    the controller decides what to do about an invalid estimate."""

    x: float  # mm, plate frame
    y: float  # mm
    vx: float  # mm/s
    vy: float  # mm/s
    t: float  # s, the timestamp this estimate refers to
    valid: bool
    cov: np.ndarray | None = None  # 4x4, diagnostics only


# --- from simulation/plate_sim.py -------------------------------------------


@dataclass
class SimParams:
    g: float = 9810.0  # mm/s^2  (note: mm units!)
    friction: float = 0.5  # viscous damping (1/s)
    plate_radius: float = 100.0  # mm; ball falls off beyond this
    # Rolling factor: for a solid sphere, effective accel = (5/7)*g*sin(theta).
    # For a ping pong ball (thin shell): (3/5)*g*sin(theta).
    rolling_factor: float = 3.0 / 5.0
    measurement_noise_std: float = 0.5  # mm, gaussian noise on measured pos
    actuator_lag_tau: float = 0.04  # s, first-order lag on plate angles
    detection_dropout_prob: float = 0.0  # chance per step the ball is "lost"


@dataclass
class SimState:
    x: float = 30.0
    y: float = -20.0
    vx: float = 0.0
    vy: float = 0.0
    roll: float = 0.0  # actual plate roll (after lag)
    pitch: float = 0.0  # actual plate pitch (after lag)
    on_plate: bool = True
