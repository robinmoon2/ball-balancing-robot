"""Per-tick run log: ONE CSV file, wiped and rewritten on every run.

Layout of the file:

    # run_start: 2026-11-24 19:06:05
    # control_mode: PID
    # controller: {"max_tilt_rad": 0.26, "x_kp": 0.0012, ...}
    # estimation: {"process_noise_std": 1500.0, ...}
    # calibration: {"origin_px": [320, 240], "mm_per_px": 0.14, ...}
    t,dt,dt_detection,...                 <- CSV header
    0.000,0.033,...                       <- one row per loop tick

The `#` lines hold everything that is constant over the run (controller
coefficients, Kalman tuning, calibration); the rows hold everything that
changes per tick. Read it with:

    pandas.read_csv("run_log.csv", comment="#")

Empty cells mean "not available this tick" (no ball detected, no Kalman
update, controller not computed because the estimate was invalid...).

Column groups:
  timing   t, dt, dt_detection, dt_estimation, dt_control, dt_actuation
  det_*    raw detector output in PIXELS: found, px_x, px_y, area, score
           (score = mean orangeness of the blob, the detection confidence;
           always 1.0 with the YUV detector, which has no score)
  meas_*   the detection converted to plate-frame mm (Kalman input)
  kf_*     Kalman estimate: valid, x, y, vx, vy, the diagonal of P
           (var_*), the innovation (measurement - prediction) and the
           number of consecutive missed frames
  ctrl_*   controller: position error, its internal terms (P/I/D parts for
           PID, position/velocity parts for LQR), the final roll/pitch
           command, whether the tilt clamp kicked in, whether it was applied
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from src.control.controller import Controller
from src.utils import BallEstimate, Calibration, Detection, PlateCommand

DEFAULT_LOG_PATH = "run_log.csv"


def _fmt(v) -> str:
    """Compact cell formatting; None / NaN -> empty cell."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return str(v)


class RunLogger:
    BASE_COLUMNS = (
        "t", "dt", "dt_detection", "dt_estimation", "dt_control", "dt_actuation",
        "det_found", "det_px_x", "det_px_y", "det_area", "det_score",
        "meas_x_mm", "meas_y_mm",
        "kf_valid", "kf_x_mm", "kf_y_mm", "kf_vx", "kf_vy",
        "kf_var_x", "kf_var_y", "kf_var_vx", "kf_var_vy",
        "kf_innov_x", "kf_innov_y", "kf_misses",
        "ctrl_ex", "ctrl_ey",
    )
    TAIL_COLUMNS = ("ctrl_roll", "ctrl_pitch", "ctrl_saturated", "applied")

    def __init__(
        self,
        controller: Controller,
        path: str | Path = DEFAULT_LOG_PATH,
        metadata: dict | None = None,
    ):
        self.path = Path(path)
        self.controller = controller
        self._term_cols = tuple(f"ctrl_{n}" for n in controller.TERM_NAMES)
        self.columns = self.BASE_COLUMNS + self._term_cols + self.TAIL_COLUMNS

        # Delete the previous run's log, then start a fresh one.
        self.path.unlink(missing_ok=True)
        self._file = open(self.path, "w", newline="")

        header = {
            "run_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "control_mode": type(controller).__name__,
            "controller": controller.coefficients(),
            **(metadata or {}),
        }
        for key, value in header.items():
            if not isinstance(value, str):
                value = json.dumps(value, default=self._json_default)
            self._file.write(f"# {key}: {value}\n")

        self._writer = csv.writer(self._file)
        self._writer.writerow(self.columns)
        self._t0: float | None = None

    @staticmethod
    def _json_default(o):
        if hasattr(o, "tolist"):  # numpy arrays / scalars
            return o.tolist()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return str(o)

    @staticmethod
    def calibration_metadata(cal: Calibration) -> dict:
        return asdict(cal)

    def log(
        self,
        t: float,
        dt: float,
        timings: dict[str, float],
        detection: Detection,
        measurement_mm: tuple[float, float] | None,
        estimate: BallEstimate,
        innovation: tuple[float, float] | None,
        misses: int,
        command: PlateCommand,
        applied: bool,
    ) -> None:
        if self._t0 is None:
            self._t0 = t

        found = detection.found
        meas = measurement_mm or (None, None)
        innov = innovation or (None, None)
        var = (
            [float(estimate.cov[i, i]) for i in range(4)]
            if estimate.cov is not None else [None] * 4
        )
        err = self.controller.last_error or (None, None)
        terms = self.controller.last_terms

        row = [
            t - self._t0, dt,
            timings.get("detection"), timings.get("estimation"),
            timings.get("control"), timings.get("actuation"),
            found,
            detection.x if found else None,
            detection.y if found else None,
            detection.area if found else None,
            detection.score if found else None,
            meas[0], meas[1],
            estimate.valid, estimate.x, estimate.y, estimate.vx, estimate.vy,
            *var,
            innov[0], innov[1], misses,
            err[0], err[1],
            *(terms.get(n) for n in self.controller.TERM_NAMES),
            command.roll, command.pitch, self.controller.saturated, applied,
        ]
        self._writer.writerow([_fmt(v) for v in row])

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
