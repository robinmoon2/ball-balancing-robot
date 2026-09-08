#!/usr/bin/env python3
"""Hardware test: plate stays flat, moves straight up and down.

Keeps the plate normal fixed at n=(0,0,1) (horizontal) and sweeps only the
center height h around H_CENTER, driving the real servos through the full
IK chain each step: solve_end_effector_positions -> solve_bearing_positions
-> solve_servo_angles -> Arm.set_angle. Useful as a hardware sanity check
before wiring in vision/control - if the plate doesn't stay level while
moving up and down, the per-arm geometry (L1/L2/L3, mount angles) or the
servo offsets are wrong.

Run from the repo root:
    .venv/bin/python3 -m src.control.test_vertical_motion
"""

from __future__ import annotations

import time

import numpy as np

from Actuator.Servo import Servo, create_pca
from control.arm import Arm
from control.inverse_kinematics import (
    PlateOrientation,
    solve_bearing_positions,
    solve_end_effector_positions,
    solve_servo_angles,
)

L = 110.0  # plate half-width: center -> spherical joint (mm)
L1 = 95.0  # distal link: elbow -> spherical joint (mm)
L2 = 70.0  # proximal link: motor axis -> elbow (mm)
L3 = 90.0  # base radius: center -> motor axis (mm)

h = 60.0  # global: starting/center plate height (mm) - plate begins here
AMPLITUDE = 40.0  # mm above/below h that the plate moves.
PERIOD_S = 4.0  # s, time for one full up/down cycle
STEP_DT = 0.00005  # s, time between commands

OFFSET_ARM_1 = 0.66  # rad, from calibration_servo.py
OFFSET_ARM_2 = 0.55
OFFSET_ARM_3 = 0.8

# Physical mount azimuth of each named arm (bras_1/2/3, matching
# calibration_servo.py) - NOT in numeric order: arm 2 sits at 0 deg,
# arm 3 at 120 deg, arm 1 at 240 deg.
MOUNT_ANGLE_ARM_1 = np.radians(240.0)
MOUNT_ANGLE_ARM_2 = np.radians(0.0)
MOUNT_ANGLE_ARM_3 = np.radians(120.0)


def build_arms(pca) -> np.ndarray:
    arm_1 = Arm(
        servo=Servo(pca, channel=0, name="bras_1", reverse=True, offset_rad=OFFSET_ARM_1),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_1,
    )
    arm_2 = Arm(
        servo=Servo(pca, channel=1, name="bras_2", reverse=True, offset_rad=OFFSET_ARM_2),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_2,
    )
    arm_3 = Arm(
        servo=Servo(pca, channel=2, name="bras_3", reverse=True, offset_rad=OFFSET_ARM_3),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_3,
    )
    return np.array([arm_1, arm_2, arm_3])


def solve_angles_for_height(arms: np.ndarray, height: float) -> np.ndarray:
    """Flat plate (n=(0,0,1)) at the given center height -> 3 joint angles."""
    orientation = PlateOrientation(n=(0.0, 0.0, 1.0), h=height)
    end_effectors = solve_end_effector_positions(orientation, arms, L)
    bearing_positions = solve_bearing_positions(orientation, arms, end_effectors, L)
    return solve_servo_angles(arms, bearing_positions)


def angles_out_of_range(arms: np.ndarray, angles: np.ndarray) -> list[str]:
    """Names of arms whose computed angle is non-finite or outside that
    arm's own calibrated [range_min, range_max] - checked here (rather
    than relying on Servo.set_angle's internal clamp) because clamping
    silently drives the arm to a mechanical limit instead of refusing."""
    if not np.isfinite(angles).all():
        return [arm.servo.name for arm, a in zip(arms, angles) if not np.isfinite(a)]
    return [
        arm.servo.name
        for arm, angle in zip(arms, angles)
        if not (arm.servo.range_min <= angle <= arm.servo.range_max)
    ]


def move_to_height(arms: np.ndarray, height: float) -> np.ndarray | None:
    """Solve and command all arms for `height`. Returns the commanded
    angles, or None (and commands nothing) if any arm's angle is
    unreachable or out of its servo range."""
    angles = solve_angles_for_height(arms, height)
    bad = angles_out_of_range(arms, angles)
    if bad:
        print(f"height={height:.1f} mm: {bad} unreachable/out of servo range, not moving")
        return None
    for arm, angle in zip(arms, angles):
        arm.set_angle(float(angle))
    return angles


def main() -> None:
    pca = create_pca()
    arms = build_arms(pca)

    try:
        print(f"Starting at h={h:.1f} mm, sweeping +/-{AMPLITUDE:.1f} mm, flat plate.")
        move_to_height(arms, h)
        time.sleep(1.0)

        t0 = time.time()
        while True:
            t = time.time() - t0
            height = h + AMPLITUDE * np.sin(2 * np.pi * t / PERIOD_S)

            angles = move_to_height(arms, height)
            if angles is not None:
                print(f"height={height:+.1f} mm  angles(deg)=" + ", ".join(f"{np.degrees(a):+.1f}" for a in angles))
            time.sleep(STEP_DT)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        print("Returning to neutral height and releasing servos...")
        try:
            move_to_height(arms, h)
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"Could not return to neutral cleanly: {e}")
        for arm in arms:
            arm.servo.release()
        pca.deinit()
        print("Done.")


if __name__ == "__main__":
    main()
