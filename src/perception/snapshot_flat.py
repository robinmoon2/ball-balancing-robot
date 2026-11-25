#!/usr/bin/env python3
"""Put the plate at its calibrated flat pose, wait, then save a camera snapshot.

Use it for a ruler/grid calibration: the servos go to the OFFSET_ARM_* values
from .env (the same neutral main.py starts from), the camera opens with the
same width/height/framerate as main.py, and after a settle delay one frame is
saved. The servos keep holding the pose until the snapshot is written, then
go limp.

Two images are written:
  <name>.png       raw frame, use this for measuring
  <name>_grid.png  same frame with a pixel grid and the image centre marked,
                   to read pixels off by eye

Nothing else may hold the camera or the servos while this runs: stop main.py.

Run from the repo root:
    UV_ENV_FILE=.env uv run -m src.perception.snapshot_flat
    UV_ENV_FILE=.env uv run -m src.perception.snapshot_flat --wait 10 --out ruler
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import cv2
import numpy as np
from dotenv import load_dotenv

from src.actuator.actuation import Actuation
from src.actuator.Servo import Servo, create_pca
from src.control.arm import Arm
from src.perception.camera import Camera

GRID_STEP_PX = 50


def env_float(name: str) -> float:
    value = os.getenv(name)
    if value is None:
        raise SystemExit(f"{name} is missing from .env (run with UV_ENV_FILE=.env)")
    return float(value)


def build_arms(pca) -> np.ndarray:
    """Same arms as main.py: channel i, reverse=True, OFFSET_ARM_i from .env."""
    arms = []
    for i in (1, 2, 3):
        arms.append(Arm(
            servo=Servo(pca, channel=i - 1, name=f"bras_{i}", reverse=True,
                        offset_rad=env_float(f"OFFSET_ARM_{i}")),
            L1=env_float("L1"), L2=env_float("L2"), L3=env_float("L3"),
            mount_angle_rad=env_float(f"MOUNT_ANGLE_ARM_{i}"),
        ))
    return np.array(arms)


def annotate(img: np.ndarray) -> np.ndarray:
    """Pixel grid + labels, image centre cross, and the robot's +X / +Y.

    +X points toward the arm at mount angle 0 and +Y toward 90 deg
    (inverse_kinematics.py). Their direction IN THE IMAGE depends on how the
    camera is mounted, so they are NOT drawn as arrows here - check them
    physically (see the printout).
    """
    out = img.copy()
    h, w = out.shape[:2]
    grid = (0, 255, 255)
    for x in range(0, w, GRID_STEP_PX):
        cv2.line(out, (x, 0), (x, h - 1), grid, 1)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, grid, 1)
    for y in range(0, h, GRID_STEP_PX):
        cv2.line(out, (0, y), (w - 1, y), grid, 1)
        cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, grid, 1)
    cx, cy = w // 2, h // 2
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(out, f"image centre ({cx},{cy})", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait", type=float, default=10.0,
                        help="seconds to hold the flat pose before the snapshot (default 10)")
    parser.add_argument("--out", default="calib_snapshot",
                        help="output file name without extension (default calib_snapshot)")
    args = parser.parse_args()

    load_dotenv()
    width = int(env_float("CAMERA_WIDTH"))
    height = int(env_float("CAMERA_HEIGHT"))
    framerate = int(env_float("CAMERA_FRAMERATE"))

    pca = create_pca()
    # Constructing Actuation writes every arm's offset_rad, i.e. the flat pose.
    actuator = Actuation(arms=build_arms(pca), plate_radius=env_float("L"),
                         neutral_height=env_float("H"), max_rate_rad_s=None)
    try:
        # Open the camera now so auto exposure / white balance / focus settle
        # during the wait, not after it.
        with Camera(source=int(env_float("CAMERA_SOURCE")), width=width,
                    height=height, framerate=framerate) as cam:
            print(f"Plate at flat pose. Place the ruler; snapshot in {args.wait:.0f} s.")
            deadline = time.monotonic() + args.wait
            last_shown = None
            while (left := deadline - time.monotonic()) > 0:
                cam.read_frame(timeout_s=0.2)  # keep the pipe drained
                if last_shown != int(left):
                    last_shown = int(left)
                    print(f"  {last_shown + 1:2d} s", end="\r", flush=True)

            frame = None
            for _ in range(10):
                frame = cam.read()
                if frame is not None:
                    break
            if frame is None:
                raise SystemExit("\nNo frame from the camera.")
    finally:
        actuator.release()

    raw_path = Path(f"{args.out}.png")
    grid_path = Path(f"{args.out}_grid.png")
    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(grid_path), annotate(frame))

    h, w = frame.shape[:2]
    print(f"\nSaved {raw_path} and {grid_path} ({w}x{h}). Servos released.")
    print(
        "\nNext:\n"
        "  1. In the image, read the pixel of the plate CENTRE  -> origin_px\n"
        "  2. Read the pixel distance between two ruler marks D mm apart\n"
        "     -> mm_per_px = D / pixels  (use marks far apart, e.g. 100 mm)\n"
        "  3. Flips: +X points to the arm with MOUNT_ANGLE 0 (bras_2), +Y is 90 deg\n"
        "     counter-clockwise from it seen from ABOVE. If that direction goes\n"
        "     toward smaller pixel x in the image, flip_x=True; if +Y goes toward\n"
        "     smaller pixel y (up in the image), flip_y=True."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
