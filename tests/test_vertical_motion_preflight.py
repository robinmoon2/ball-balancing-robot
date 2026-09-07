"""Hardware preflight check for src/control/test_vertical_motion.py.

Runs the real IK chain with the script's actual CONFIG values, WITHOUT
touching any hardware (arms are built with pca=None; nothing ever calls
Servo.set_angle), and checks every computed angle actually lands inside
that arm's calibrated servo range. Run this BEFORE running
test_vertical_motion.py on the real rig - it's exactly the check that
would have caught the arm slamming into its mechanical limit:

Diagnosis of the "arm goes too high, plate deformed" report:
    solve_servo_angles returns the RAW geometric joint angle (atan2 of the
    elbow position relative to the base pivot), with no adjustment for
    where "flat" happens to land for this arm's L1/L2/L3/L geometry. For
    the CONFIG values in test_vertical_motion.py, that raw angle comes out
    near +-180 degrees even at the flat, centered starting height h - but
    a servo calibrated via calibration_servo.py (offset_rad ~ 0.55-0.8 rad,
    i.e. ~31-46 deg) only has roughly +-40 to +140 degrees of *commandable*
    range around its own zero. Servo.set_angle clips anything outside
    [0, pi] raw to the mechanical limit, so every arm was driven straight
    to (a different) hard stop instead of tracking the small height
    change - hence "too high" and a twisted/deformed plate (each arm
    clips by a different amount because each has a different offset_rad).

The fix isn't in this test file - it's deciding how solve_servo_angles /
solve_bearing_positions should reference "flat" (or re-deriving L1/L2/L3/L
so flat naturally lands near 0). This file just gives a repeatable way to
check that before running on hardware again:

    .venv/bin/python3 tests/test_vertical_motion_preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.control import test_vertical_motion as tvm


def check_angles_in_range(height: float) -> list[str]:
    """Return a list of human-readable problems for this height, empty if OK."""
    arms = tvm.build_arms(pca=None)
    angles = tvm.solve_angles_for_height(arms, height)

    problems = []
    for arm, angle in zip(arms, angles):
        if not np.isfinite(angle):
            problems.append(f"{arm.servo.name} at h={height:.4f}: non-finite angle")
            continue
        if not (arm.servo.range_min <= angle <= arm.servo.range_max):
            problems.append(
                f"{arm.servo.name} at h={height:.4f}: angle "
                f"{np.degrees(angle):+.1f} deg outside servo range "
                f"[{np.degrees(arm.servo.range_min):+.1f}, "
                f"{np.degrees(arm.servo.range_max):+.1f}] deg "
                f"(offset_rad={arm.servo.offset_rad:.3f})"
            )
    return problems


def test_neutral_height_within_servo_range():
    """At the configured neutral height h (flat plate), every arm's angle
    must land inside its own calibrated [range_min, range_max]."""
    problems = check_angles_in_range(tvm.h)
    assert not problems, "\n" + "\n".join(problems)


def test_full_sweep_within_servo_range():
    """Same check, across the whole h +/- AMPLITUDE range the live loop
    actually commands."""
    problems = []
    for height in np.linspace(tvm.h - tvm.AMPLITUDE, tvm.h + tvm.AMPLITUDE, 11):
        problems += check_angles_in_range(height)
    assert not problems, "\n" + "\n".join(problems)


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
