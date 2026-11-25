#!/usr/bin/env python3
"""Measure the chroma bounds OrangeDetector.detect_yuv needs.

detect_yuv is ~100x cheaper than detect(), but it thresholds raw U/V
instead of computing the orangeness score, so its bounds depend on your
ball and your lighting. This derives them the safe way: it runs the
TRUSTED BGR detector on live frames, takes the chroma pixels that landed
inside the blob it found, and reports the box that contains them.

It also reports what fraction of the frame the resulting box would accept
overall - if that is much larger than the ball, something else on the
plate shares the ball's chroma and you should keep using detect().

Put the ball on the plate, under the lighting you will actually run in,
and move it around a bit so a few positions are sampled.

Run from the repo root:
    .venv/bin/python3 -m src.perception.calibrate_yuv
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from camera import Camera
from detector import OrangeDetector

SAMPLES = 40
TIMEOUT_S = 30.0
MARGIN = 8  # widen the measured box by this many levels, for lighting drift
KEEP = 8  # sampled frames retained to validate the bounds against


def main() -> None:
    cam = Camera(source=0, width=640, height=480)
    det = OrangeDetector(min_area=200, score_threshold=0.15)

    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept: list[tuple] = []  # (frame, detection) to check the bounds against
    accepted = 0
    deadline = time.monotonic() + TIMEOUT_S

    print(f"Sampling {SAMPLES} detections (move the ball around slowly)...")
    while accepted < SAMPLES and time.monotonic() < deadline:
        frame = cam.read_frame()
        if frame is None:
            continue

        detection = det.detect(frame.bgr)
        if not detection.found:
            continue

        # Rebuild the blob as a chroma-resolution mask, then read U/V/Y
        # under it. Radius from the area, which detect() already computed.
        radius_px = max(2, int(np.sqrt(detection.area / np.pi)))
        blob = np.zeros((frame.height // 2, frame.width // 2), np.uint8)
        cv2.circle(blob, (detection.x // 2, detection.y // 2), radius_px // 2, 255, -1)
        inside = blob == 255
        if inside.sum() < 10:
            continue

        us.append(frame.u[inside])
        vs.append(frame.v[inside])
        ys.append(frame.y[::2, ::2][inside])
        kept.append((frame, detection))
        del kept[:-KEEP]
        accepted += 1
        print(f"\r  {accepted}/{SAMPLES}", end="", flush=True)

    cam.release()
    print()

    if accepted < 5:
        print("Not enough detections - is the ball in frame and lit?")
        sys.exit(1)

    u = np.concatenate(us)
    v = np.concatenate(vs)
    y = np.concatenate(ys)

    # Percentiles, not min/max: a couple of edge pixels straddle the
    # background and would blow the box wide open.
    u_max = int(np.percentile(u, 95)) + MARGIN
    v_min = int(np.percentile(v, 5)) - MARGIN
    y_min = max(0, int(np.percentile(y, 5)) - MARGIN)

    print(f"\nsamples: {accepted} detections, {u.size} chroma pixels")
    print(f"  U inside ball: median {np.median(u):5.1f}  p5 {np.percentile(u,5):5.1f}  p95 {np.percentile(u,95):5.1f}")
    print(f"  V inside ball: median {np.median(v):5.1f}  p5 {np.percentile(v,5):5.1f}  p95 {np.percentile(v,95):5.1f}")
    print(f"  Y inside ball: median {np.median(y):5.1f}  p5 {np.percentile(y,5):5.1f}  p95 {np.percentile(y,95):5.1f}")

    # Validate the bounds on frames we KNOW were good: taken during the
    # sampling loop, so auto-exposure had settled and the ball was in
    # shot. (Reopening the camera here instead would read frame 0 of a
    # fresh stream - near-black, chroma still neutral - and reject
    # everything, including the ball we just measured.)
    probe = OrangeDetector(min_area=det.min_area, u_max=u_max, v_min=v_min, y_min=y_min)
    errors: list[float] = []
    misses = 0
    fractions: list[float] = []
    for frame, reference in kept:
        box = (
            (frame.u <= u_max)
            & (frame.v >= v_min)
            & (frame.y[::2, ::2] >= y_min)
        )
        fractions.append(float(box.mean()))
        found = probe.detect_yuv(frame)
        if not found.found:
            misses += 1
        else:
            errors.append(float(np.hypot(found.x - reference.x, found.y - reference.y)))

    frac = float(np.median(fractions))
    print(f"\nchecked against {len(kept)} sampled frames:")
    print(f"  the box accepts {100*frac:.2f}% of a frame "
          f"({int(frac * 640 * 480)} full-res px; the ball is ~{det.min_area}+ px)")
    if errors:
        print(f"  detect_yuv vs detect centroid: median {np.median(errors):.1f} px, "
              f"max {max(errors):.1f} px")
    if misses:
        print(f"  MISSED the ball on {misses}/{len(kept)} frames - bounds too tight, "
              f"raise MARGIN and re-run.")
    if frac > 0.10:
        print("  WARNING: that is far more than a ball. Something else on the")
        print("  plate shares its chroma - prefer detect() until you fix the scene.")
    if errors and np.median(errors) > 10:
        print("  WARNING: the two detectors disagree on where the ball is;")
        print("  detect_yuv is latching onto something else. Prefer detect().")

    print("\nPaste into main.py:\n")
    print(f"detector = OrangeDetector(min_area=detection_min_area,")
    print(f"                          score_threshold=score_threshold,")
    print(f"                          u_max={u_max}, v_min={v_min}, y_min={y_min})")
    print("...then pass use_yuv=True to Perception(...).")


if __name__ == "__main__":
    main()
