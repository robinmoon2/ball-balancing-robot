"""Tests for src/control/inverse_kinematics.py.

No test runner is installed in .venv yet (no pytest), so this file is
plain assert-based and runnable directly:

    .venv/bin/python3 tests/test_inverse_kinematics.py

It's also pytest-discoverable as-is (test_* functions, plain asserts) if
you later add pytest as a dev dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.control.arm import Arm
from src.control.inverse_kinematics import (
    PlateOrientation,
    VectorPosition,
    incline_board,
    normal_vector_board,
    solve_bearing_positions,
    solve_end_effector_positions,
    solve_servo_angles,
)

THETAS = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)


def make_arm(theta: float, L1: float = 2.5, L2: float = 2.0, L3: float = 1.5) -> Arm:
    # servo=None: none of the functions under test touch arm.servo.
    return Arm(servo=None, L1=L1, L2=L2, L3=L3, mount_angle_rad=theta)


def make_arms(L1: float = 2.5, L2: float = 2.0, L3: float = 1.5) -> np.ndarray:
    return np.array([make_arm(th, L1, L2, L3) for th in THETAS])


# ---------- solve_end_effector_positions ----------

def test_end_effector_on_plate():
    """Every P_i must sit exactly L from the plate center C, in the plane
    through C with normal n (i.e. actually on the tilted plate)."""
    n, h, L = (0.3, -0.4, 0.866), 5.0, 2.0
    C = np.array([0.0, 0.0, h])
    orientation = PlateOrientation(n=n, h=h)

    positions = solve_end_effector_positions(orientation, make_arms(), L)
    for pos in positions:
        P = np.array([pos.x, pos.y, pos.z])
        assert abs(np.linalg.norm(P - C) - L) < 1e-9          # distance
        assert abs((P - C) @ np.array(n)) < 1e-9               # orthogonality


def test_end_effector_horizontal_along_azimuth():
    """P_i's horizontal projection must lie exactly along its own azimuth
    ray - i.e. b == 0 in the arm's local (radial, tangential, height) frame."""
    n, h, L = (0.3, -0.4, 0.866), 5.0, 2.0
    orientation = PlateOrientation(n=n, h=h)

    positions = solve_end_effector_positions(orientation, make_arms(), L)
    for pos, th in zip(positions, THETAS):
        b = -pos.x * np.sin(th) + pos.y * np.cos(th)
        assert abs(b) < 1e-9


def test_end_effector_flat_plate():
    """At n=(0,0,1) (flat), every P_i should sit at height h, at radius L
    on its own azimuth - the simple known-configuration sanity check."""
    h, L = 3.0, 2.0
    orientation = PlateOrientation(n=(0.0, 0.0, 1.0), h=h)

    positions = solve_end_effector_positions(orientation, make_arms(), L)
    for pos, th in zip(positions, THETAS):
        assert abs(pos.x - L * np.cos(th)) < 1e-9
        assert abs(pos.y - L * np.sin(th)) < 1e-9
        assert abs(pos.z - h) < 1e-9


# ---------- normal_vector_board ----------

def test_normal_vector_board_flat():
    orientation = normal_vector_board(pitch=0.0, roll=0.0, height=7.0)
    assert np.allclose(orientation.n, (0.0, 0.0, 1.0))
    assert orientation.h == 7.0


def test_normal_vector_board_unit_norm():
    """n = (sin(roll), -sin(pitch), sqrt(...)) should stay unit-length
    across the plate's whole commandable tilt range."""
    rng = np.random.default_rng(1)
    max_tilt = np.radians(15)  # matches Controller's default max_tilt_rad
    for _ in range(50):
        pitch = rng.uniform(-max_tilt, max_tilt)
        roll = rng.uniform(-max_tilt, max_tilt)
        orientation = normal_vector_board(pitch, roll, height=1.0)
        assert abs(np.linalg.norm(orientation.n) - 1.0) < 1e-9


# ---------- solve_bearing_positions ----------

def test_solve_bearing_positions_satisfies_constraints_all_arms():
    """All 3 arms (not just the one at azimuth 0) must produce a finite
    bearing position satisfying both link-length constraints, checked in
    that arm's own local (a, b, c) frame.

    Regression test for the missing local-frame rotation: solve_bearing_positions
    used to feed each end effector's raw global (x, y, z) straight into its
    2-link solve, which only happens to be correct for the arm at azimuth 0
    (global frame == local frame there) and produced NaN for the other two.
    """
    orientation = PlateOrientation(n=(0.05, -0.05, 0.99), h=4.0)
    arms = make_arms(L1=4.0, L2=4.0, L3=1.0)
    end_effectors = solve_end_effector_positions(orientation, arms, L=2.0)

    bearing_positions = solve_bearing_positions(orientation, arms, end_effectors, L=2.0)
    assert len(bearing_positions) == len(arms)

    for arm, end_effector, bearing in zip(arms, end_effectors, bearing_positions):
        assert np.isfinite([bearing.x, bearing.y, bearing.z]).all()

        theta_i = arm.get_azimuth()
        a = end_effector.x * np.cos(theta_i) + end_effector.y * np.sin(theta_i)
        b = -end_effector.x * np.sin(theta_i) + end_effector.y * np.cos(theta_i)
        c = end_effector.z

        base_dist_sq = (bearing.x - arm.L3) ** 2 + bearing.z ** 2
        target_dist_sq = (bearing.x - a) ** 2 + (bearing.y - b) ** 2 + (bearing.z - c) ** 2
        assert abs(base_dist_sq - arm.L2**2) < 1e-6
        assert abs(target_dist_sq - arm.L1**2) < 1e-6


def test_bearing_position_round_trip():
    """Known elbow config -> synthetic end-effector target -> the bearing
    position solve_bearing_positions recovers must satisfy both link-length
    constraints (base-to-elbow = L2, elbow-to-target = L1), across elbow
    angles spanning all four quadrants.
    """
    rng = np.random.default_rng(0)
    dummy_orientation = PlateOrientation(n=(0.0, 0.0, 1.0), h=0.0)

    checked = 0
    for q in np.linspace(-np.pi + 0.2, np.pi - 0.2, 13):
        L1, L2, L3 = 1.7, 2.3, 1.1
        mx, mz = L3 + L2 * np.cos(q), L2 * np.sin(q)

        phi = rng.uniform(0, 2 * np.pi)
        px, pz = mx + L1 * np.cos(phi), mz + L1 * np.sin(phi)
        if abs(pz) < 0.2:
            continue  # solve_bearing_positions divides by end_effector.z

        arm = make_arm(theta=0.0, L1=L1, L2=L2, L3=L3)
        end_effector = VectorPosition(x=px, y=0.0, z=pz)

        bearing_positions = solve_bearing_positions(
            dummy_orientation, np.array([arm]), np.array([end_effector]), L=1.0
        )
        b = bearing_positions[0]

        base_dist_sq = (b.x - L3) ** 2 + b.z ** 2
        target_dist_sq = (b.x - px) ** 2 + (b.y - 0.0) ** 2 + (b.z - pz) ** 2
        assert abs(base_dist_sq - L2**2) < 1e-6
        assert abs(target_dist_sq - L1**2) < 1e-6
        checked += 1

    assert checked > 5  # sanity: the |pz| guard didn't skip everything


# ---------- solve_servo_angles ----------

def test_solve_servo_angles_recovers_known_angle():
    """Place a bearing at a known joint angle q (bearing.x = L3 + L2*cos(q),
    bearing.z = L2*sin(q)) and check solve_servo_angles recovers q, across
    all four quadrants - catches a missing-parentheses / precedence bug in
    cos_theta's formula (currently: `bearing.x - arm.L3 / arm.L2`, which
    Python evaluates as `bearing.x - (arm.L3 / arm.L2)` instead of the
    intended `(bearing.x - arm.L3) / arm.L2`).
    """
    L2, L3 = 2.0, 1.5
    arms = np.array([make_arm(theta=0.0, L2=L2, L3=L3)])

    for q_true in np.linspace(-np.pi + 0.1, np.pi - 0.1, 9):
        bearing = VectorPosition(
            x=L3 + L2 * np.cos(q_true), y=0.0, z=L2 * np.sin(q_true)
        )
        angles = solve_servo_angles(arms, np.array([bearing]))
        diff = np.angle(np.exp(1j * (angles[0] - q_true)))  # wrap to [-pi, pi]
        assert abs(diff) < 1e-9, f"q_true={q_true}, recovered={angles[0]}"


# ---------- incline_board (full chain) ----------

def test_incline_board_returns_one_angle_per_arm():
    # incline_board uses the module-level L=5 plate radius, so these arms
    # need enough reach (L1+L2) to actually span base -> target at that
    # radius and height - make_arms()'s smaller defaults are unreachable
    # here and would trip solve_bearing_positions's domain check.
    correction = SimpleNamespace(
        pitch_rad=np.radians(5.0),
        roll_rad=np.radians(-3.0),
        height_m=4.0,
        arms=make_arms(L1=4.0, L2=4.0, L3=1.0),
    )
    angles = incline_board(correction)
    assert len(angles) == len(correction.arms)
    assert all(np.isfinite(angles))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        sys.exit(1)
