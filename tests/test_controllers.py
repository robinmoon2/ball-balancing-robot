"""Tests for the Controller base class and its PID / LQR subclasses.

Plain assert-based, runnable directly (no hardware touched):

    .venv/bin/python3 tests/test_controllers.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.control.controller import Controller
from src.control.controller_lqr import ControllerLQR
from src.control.controller_pid import ControllerPID
from src.utils import PID, BallEstimate, StateSpaceModel

MAX_TILT = math.radians(20)
G_EFF = 7000.0


def make_lqr() -> ControllerLQR:
    return ControllerLQR.from_bryson(StateSpaceModel(g_eff=G_EFF), 50.0, 300.0, MAX_TILT)


def make_pid() -> ControllerPID:
    return ControllerPID(PID(kp=0.0012, ki=0.0002, kd=0.0009),
                         PID(kp=0.0012, ki=0.0002, kd=0.0009), MAX_TILT)


def est(x=0.0, y=0.0, vx=0.0, vy=0.0, valid=True) -> BallEstimate:
    return BallEstimate(x=x, y=y, vx=vx, vy=vy, t=0.0, valid=valid)


def test_both_are_controllers():
    assert isinstance(make_pid(), Controller)
    assert isinstance(make_lqr(), Controller)


def test_lqr_gain_stabilizes_model():
    c = make_lqr()
    assert c.K.shape == (2, 4) and np.all(np.isfinite(c.K))
    m = c.model
    eig = np.linalg.eigvals(m.A - m.B @ c.K)
    assert np.all(eig.real < 0), eig


def test_signs_agree_between_pid_and_lqr():
    for c in (make_pid(), make_lqr()):
        cmd = c.update(est(x=20.0), 1 / 30)
        assert cmd.pitch < 0, (type(c).__name__, cmd)
        c.reset()
        cmd = c.update(est(y=20.0), 1 / 30)
        assert cmd.roll > 0, (type(c).__name__, cmd)


def test_invalid_estimate_holds_flat_and_resets():
    c = make_pid()
    c.update(est(x=30.0), 1 / 30)
    assert c.pid_x._integral != 0.0
    cmd = c.update(est(x=30.0, valid=False), 1 / 30)
    assert cmd.roll == 0.0 and cmd.pitch == 0.0
    assert c.pid_x._integral == 0.0
    assert make_lqr().update(est(x=30.0, valid=False), 1 / 30).pitch == 0.0


def test_tilt_clamped():
    for c in (make_pid(), make_lqr()):
        cmd = c.update(est(x=1e6, y=1e6), 1 / 30)
        assert math.hypot(cmd.roll, cmd.pitch) <= MAX_TILT + 1e-12


def test_non_positive_dt_holds_last_command():
    c = make_lqr()
    first = c.update(est(x=10.0), 1 / 30)
    assert c.update(est(x=-50.0), 0.0) == first


def test_lqr_closed_loop_sim_converges():
    """Euler-integrate the same linear model the gain was designed on."""
    c = make_lqr()
    dt = 1 / 30
    x, y, vx, vy = 30.0, -20.0, 0.0, 0.0
    for _ in range(int(5.0 / dt)):
        cmd = c.update(est(x, y, vx, vy), dt)
        ax = G_EFF * cmd.pitch
        ay = -G_EFF * cmd.roll
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
    assert math.hypot(x, y) < 2.0, (x, y)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
