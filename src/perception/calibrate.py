#!/usr/bin/env python3
"""Interactive camera calibration: solve Calibration(origin_px, mm_per_px).

Place the ball at known points on the plate; this reads back where the
detector sees it in pixels and solves the pixel -> plate-mm mapping, then
prints the exact Calibration(...) line to paste into main.py.

You need at minimum two placements: the plate CENTRE, and one point a
measured distance away along your +X axis. A third along +Y is optional but
lets it verify the scale instead of assuming both axes match, and measure
the y flip rather than guessing it (this rig's camera looks UP at the plate
from below, so at least one axis is probably mirrored).

Nothing else may hold the camera while this runs - stop main.py first.

Run from the repo root:
    .venv/bin/python3 -m src.perception.calibrate
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.perception.camera import Camera
from src.perception.detector import OrangeDetector

SAMPLES = 15  # detections to median together per placement
TIMEOUT_S = 20.0


def capture(cam: Camera, detector: OrangeDetector, label: str) -> tuple[float, float]:
    """Median pixel position over SAMPLES detections - steadier than one frame."""
    xs: list[int] = []
    ys: list[int] = []
    deadline = time.monotonic() + TIMEOUT_S

    print(f"  reading {label} ...", end="", flush=True)
    while len(xs) < SAMPLES:
        if time.monotonic() > deadline:
            print()
            raise SystemExit(
                f"Timed out: only {len(xs)}/{SAMPLES} detections in {TIMEOUT_S:.0f}s.\n"
                "The detector isn't seeing the ball. Check lighting, or lower "
                "score_threshold / min_area in OrangeDetector."
            )
        frame = cam.read()
        if frame is None:
            continue
        d = detector.detect(frame)
        if d.found:
            xs.append(d.x)
            ys.append(d.y)
            print(".", end="", flush=True)

    px, py = float(np.median(xs)), float(np.median(ys))
    spread = max(np.std(xs), np.std(ys))
    print(f" pixel=({px:.1f}, {py:.1f})  jitter={spread:.1f}px")
    if spread > 5.0:
        print(f"  ! noisy detection ({spread:.1f}px scatter) - calibration will inherit that error")
    return px, py


def ask_distance(axis: str) -> float | None:
    raw = input(f"  measured distance from centre along +{axis}, in mm "
                f"(Enter to skip): ").strip()
    if not raw:
        return None
    try:
        d = float(raw)
    except ValueError:
        raise SystemExit(f"Not a number: {raw!r}")
    if d == 0.0:
        raise SystemExit("Distance must be non-zero.")
    return abs(d)


def main() -> None:
    detector = OrangeDetector(min_area=200, score_threshold=0.15)

    with Camera() as cam:
        print("Calibration: place the ball as prompted, then press Enter.\n")

        input("1. Ball at the plate CENTRE. Press Enter...")
        ox, oy = capture(cam, detector, "centre")

        print("\n2. Ball at a measured distance along your +X axis.")
        dist_x = ask_distance("X")
        if dist_x is None:
            raise SystemExit("The +X point is required - nothing to solve from.")
        input("   Press Enter when it's in place...")
        x1, y1 = capture(cam, detector, "+X point")

        print("\n3. Optional: ball at a measured distance along your +Y axis.")
        dist_y = ask_distance("Y")
        if dist_y is not None:
            input("   Press Enter when it's in place...")
            x2, y2 = capture(cam, detector, "+Y point")
        else:
            x2 = y2 = None

    # --- solve -------------------------------------------------------------
    dx = x1 - ox
    dy = y1 - oy
    travel_x = float(np.hypot(dx, dy))
    if travel_x < 5.0:
        raise SystemExit(
            f"The +X point moved only {travel_x:.1f}px from centre - too small to "
            "calibrate. Use a larger distance."
        )
    scale_x = dist_x / travel_x

    # to_mm negates when flip_* is set, so pick the flag that makes the
    # measured move come out POSITIVE along the axis the ball actually moved.
    flip_x = dx < 0

    scale = scale_x
    flip_y = True  # default: image rows grow down, plate +Y is up
    if x2 is not None:
        dx2 = x2 - ox
        dy2 = y2 - oy
        travel_y = float(np.hypot(dx2, dy2))
        scale_y = dist_y / travel_y
        flip_y = dy2 < 0
        disagreement = abs(scale_x - scale_y) / max(scale_x, scale_y) * 100
        print(f"\n  scale from +X point: {scale_x:.4f} mm/px")
        print(f"  scale from +Y point: {scale_y:.4f} mm/px  ({disagreement:.1f}% apart)")
        if disagreement > 10:
            print("  ! >10% apart. The camera may not be square-on to the plate, or a "
                  "measured distance is off. A single mm_per_px will be approximate.")
        scale = (scale_x + scale_y) / 2.0

    print("\n" + "=" * 62)
    print("Paste this into main.py:\n")
    print(f"    Calibration(")
    print(f"        origin_px=({ox:.1f}, {oy:.1f}),")
    print(f"        mm_per_px={scale:.5f},")
    print(f"        flip_x={flip_x},")
    print(f"        flip_y={flip_y},")
    print(f"    )")
    print("=" * 62)

    # sanity check: round-trip the points we measured
    from src.utils import Calibration

    cal = Calibration(origin_px=(ox, oy), mm_per_px=scale, flip_x=flip_x, flip_y=flip_y)
    print("\nCheck - what this mapping reports for the points you just placed:")
    print(f"  centre   -> {tuple(round(v, 1) for v in cal.to_mm(ox, oy))} mm   (want ~(0, 0))")
    print(f"  +X point -> {tuple(round(v, 1) for v in cal.to_mm(x1, y1))} mm   (want ~({dist_x:.0f}, 0))")
    if x2 is not None:
        print(f"  +Y point -> {tuple(round(v, 1) for v in cal.to_mm(x2, y2))} mm   (want ~(0, {dist_y:.0f}))")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
