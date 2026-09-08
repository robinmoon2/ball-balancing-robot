#!/usr/bin/env python3
"""Hardware test: constant tilt, rotating tilt DIRECTION - the plate's low
point travels around the rim in a circle (a cone / precession), spiralling
out from flat so faults show up small before they get big.

WHY THIS TEST (it complements test_vertical_motion.py exactly):
    test_vertical_motion holds n=(0,0,1), which makes
        d_i = alpha*cos(theta_i) + beta*sin(theta_i)  ==  0
        N_i = sqrt(d_i^2 + gamma^2)                   ==  1
        z_i = h - L*d_i/N_i                           ==  h
    i.e. the entire plate-normal projection is multiplied out of existence,
    all three arms get identical angles, and the azimuth assignment is
    invisible. Sweeping the tilt DIRECTION through 360 deg fixes all of that:
      - d_i sweeps its full +/- range for every arm
      - the three arms cycle 120 deg out of phase (differential motion,
        which the flat test never produces)
      - a wrong arm->azimuth mapping shows up as the wrong phase ORDER
      - a flipped sign shows up as the circle running backwards

WHAT TO WATCH:
    1. Tilt magnitude should look CONSTANT as the direction rotates - the
       plate should wobble like a coin settling, not lurch.
    2. The "lowest" arm printed each step should advance smoothly in the
       order bras_2 -> bras_3 -> bras_1 (the normal leans toward azimuth
       psi, so the arm AT psi is pushed lowest). Wrong order or reversed
       direction => the mount-angle mapping or a sign is wrong.
    3. No vertical bobbing: the plate centre should stay at H_CIRCLE.

Run from the repo root:
    .venv/bin/python3 -m src.control.test_circle_motion
    .venv/bin/python3 -m src.control.test_circle_motion --dry-run   (no hardware)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from Actuator.Servo import create_pca
from control.inverse_kinematics import (
    PlateOrientation,
    solve_bearing_positions,
    solve_end_effector_positions,
    solve_servo_angles,
)
# Rig geometry and arm construction are shared with the vertical test so
# there is one source of truth for the measured lengths.
from control.test_vertical_motion import L, angles_out_of_range, build_arms

### CONFIG ###
H_CIRCLE = 90.0  # mm, plate centre height held constant through the circle.
TILT_DEG = 20.0  # cone half-angle: how far the normal leans from vertical
REVOLUTION_S = 1.0  # s for the tilt direction to make one full turn
RAMP_REVOLUTIONS = 2.0  # turns spent ramping tilt 0 -> TILT_DEG (the "spiral")
STEP_DT = 0.005  # s between commands

ARM_NAMES = ("bras_1", "bras_2", "bras_3")


def orientation_for(tilt_rad: float, azimuth_rad: float, height: float) -> PlateOrientation:
    """Plate normal leaning `tilt_rad` from vertical, toward `azimuth_rad`."""
    n = (
        np.sin(tilt_rad) * np.cos(azimuth_rad),
        np.sin(tilt_rad) * np.sin(azimuth_rad),
        np.cos(tilt_rad),
    )
    return PlateOrientation(n=n, h=height)


def solve_angles(arms: np.ndarray, tilt_rad: float, azimuth_rad: float, height: float) -> np.ndarray:
    orientation = orientation_for(tilt_rad, azimuth_rad, height)
    end_effectors = solve_end_effector_positions(orientation, arms, L)
    bearing_positions = solve_bearing_positions(orientation, arms, end_effectors, L)
    return solve_servo_angles(arms, bearing_positions)


def tilt_at(t: float) -> float:
    """Cone angle at time t - ramps 0 -> TILT_DEG over RAMP_REVOLUTIONS."""
    ramp_s = RAMP_REVOLUTIONS * REVOLUTION_S
    frac = 1.0 if ramp_s <= 0 else min(1.0, t / ramp_s)
    return np.radians(TILT_DEG) * frac


def dry_run(arms: np.ndarray) -> bool:
    """Walk the whole planned trajectory offline. Returns True if every pose
    is reachable and in range. Nothing is commanded."""
    total_s = (RAMP_REVOLUTIONS + 1.0) * REVOLUTION_S
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    problems = []

    for t in np.arange(0.0, total_s, STEP_DT):
        tilt = tilt_at(t)
        az = 2 * np.pi * t / REVOLUTION_S
        angles = solve_angles(arms, tilt, az, H_CIRCLE)
        bad = angles_out_of_range(arms, angles)
        if bad:
            problems.append(
                f"  t={t:5.2f}s tilt={np.degrees(tilt):4.1f} az={np.degrees(az) % 360:5.1f}: {bad}"
            )
        if np.isfinite(angles).all():
            lo = np.minimum(lo, angles)
            hi = np.maximum(hi, angles)

    print(f"Dry run: tilt 0 -> {TILT_DEG} deg at h={H_CIRCLE} mm, "
          f"{RAMP_REVOLUTIONS + 1:.0f} revolutions, {total_s:.1f} s")
    for i, arm in enumerate(arms):
        s = arm.servo
        print(f"  {ARM_NAMES[i]}: q {np.degrees(lo[i]):+7.2f} .. {np.degrees(hi[i]):+7.2f} deg   "
              f"servo range [{np.degrees(s.range_min):+7.2f}, {np.degrees(s.range_max):+7.2f}]   "
              f"margin {np.degrees(min(lo[i] - s.range_min, s.range_max - hi[i])):+6.2f} deg")
    if problems:
        print(f"  {len(problems)} pose(s) out of range, first few:")
        for p in problems[:5]:
            print(p)
        return False
    print("  all poses reachable and in range.")
    return True


def main(dry: bool = False) -> None:
    if dry:
        arms = build_arms(pca=None)  # no hardware touched
        sys.exit(0 if dry_run(arms) else 1)

    pca = create_pca()
    arms = build_arms(pca)

    # Always validate the whole trajectory before moving anything.
    if not dry_run(arms):
        print("Refusing to run: trajectory leaves the servo range.")
        for arm in arms:
            arm.servo.release()
        pca.deinit()
        return

    try:
        print(f"\nGoing flat at h={H_CIRCLE:.1f} mm...")
        angles = solve_angles(arms, 0.0, 0.0, H_CIRCLE)
        for arm, angle in zip(arms, angles):
            arm.set_angle(float(angle))
        time.sleep(1.5)

        print("Spiralling out. Ctrl+C to stop.\n")
        t0 = time.time()
        while True:
            t = time.time() - t0
            tilt = tilt_at(t)
            az = 2 * np.pi * t / REVOLUTION_S

            angles = solve_angles(arms, tilt, az, H_CIRCLE)
            bad = angles_out_of_range(arms, angles)
            if bad:
                print(f"tilt={np.degrees(tilt):4.1f} az={np.degrees(az) % 360:5.1f}: "
                      f"{bad} out of range, skipping")
                time.sleep(STEP_DT)
                continue

            for arm, angle in zip(arms, angles):
                arm.set_angle(float(angle))

            lowest = ARM_NAMES[int(np.argmin(angles))]
            print(f"tilt={np.degrees(tilt):4.1f} deg  az={np.degrees(az) % 360:5.1f} deg  "
                  f"q(deg)=" + ", ".join(f"{np.degrees(a):+6.1f}" for a in angles)
                  + f"   lowest={lowest}")
            time.sleep(STEP_DT)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        print("Returning to flat and releasing servos...")
        try:
            angles = solve_angles(arms, 0.0, 0.0, H_CIRCLE)
            if not angles_out_of_range(arms, angles):
                for arm, angle in zip(arms, angles):
                    arm.set_angle(float(angle))
                time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"Could not return to flat cleanly: {e}")
        for arm in arms:
            arm.servo.release()
        pca.deinit()
        print("Done.")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
