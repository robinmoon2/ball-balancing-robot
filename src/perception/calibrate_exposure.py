#!/usr/bin/env python3
"""Measure fixed exposure / gain / white-balance values for this rig.

WHY LOCK THEM AT ALL
    libcamera's auto-exposure and auto-white-balance are tuned to make
    pictures look good, which is the wrong objective for a control loop:

      - Startup transient. AE needs roughly half a second to converge, and
        swings the exposure several-fold on the way. Detection during that
        window is unreliable.
      - Mid-run chasing. Tilting the plate changes how much light reaches
        the sensor, so AE and AWB re-converge while you are balancing. That
        moves the U and V levels that OrangeDetector.detect_yuv thresholds
        against - the bounds you calibrated stop matching the picture.

    Locking shutter, gain and AWB gains removes both. The cost is that the
    values are only right for the lighting you measured them in.

WHAT IT DOES
    Phase 1 asks the camera what its own auto algorithms settle on, by
    reading rpicam-vid's per-frame metadata. That is the starting point -
    no guessing.

    Phase 2 runs the real Camera class twice, once with AUTO and once with
    those values LOCKED, and measures the same things both times so you can
    see what locking bought:

      - achieved frame rate, from the frames' own capture timestamps
      - detection rate with the trusted BGR detector
      - centroid scatter on a STILL ball, in pixels. This is exactly
        Estimation's `measurement_noise_std` - the script prints it.
      - U/V stability inside the ball, i.e. whether detect_yuv's bounds
        will still match in ten minutes
      - luma clipping

HOW TO RUN IT
    Put the ball on the plate and LEAVE IT STILL - the centroid scatter
    number is meaningless if the ball moves. Light the scene the way you
    will actually run. Stop main.py first; nothing else may hold the camera.

        .venv/bin/python3 src/perception/calibrate_exposure.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from camera import Camera
from detector import OrangeDetector

WIDTH, HEIGHT, FRAMERATE = 640, 480, 30
OBSERVE_S = 5.0  # phase 1: how long to let the auto algorithms settle
SAMPLES = 90  # phase 2: frames measured per configuration
BALL_SPEED_MM_S = 300.0  # for the motion-blur estimate only


# --------------------------------------------------------------------------
# Phase 1: what do the auto algorithms converge to?
# --------------------------------------------------------------------------
def observe_auto() -> dict:
    meta_path = Path(tempfile.gettempdir()) / "calib_exposure_meta.json"
    meta_path.unlink(missing_ok=True)

    print(f"Phase 1: watching auto-exposure for {OBSERVE_S:.0f} s ...")
    subprocess.run(
        [
            "rpicam-vid", "--codec", "yuv420", "--nopreview",
            "--timeout", str(int(OBSERVE_S * 1000)),
            "--width", str(WIDTH), "--height", str(HEIGHT),
            "--framerate", str(FRAMERATE),
            "--metadata", str(meta_path), "--metadata-format", "json",
            "--output", "/dev/null",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

    frames = json.loads(meta_path.read_text())
    if len(frames) < 10:
        print(f"  only {len(frames)} frames of metadata - camera busy?")
        sys.exit(1)

    exposure = np.array([f["ExposureTime"] for f in frames], float)
    gain = np.array([f["AnalogueGain"] for f in frames], float)
    red = np.array([f["ColourGains"][0] for f in frames], float)
    blue = np.array([f["ColourGains"][1] for f in frames], float)
    lux = np.array([f["Lux"] for f in frames], float)
    duration = np.array([f["FrameDuration"] for f in frames], float)

    # "Settled" = the last third, by which point AE has stopped hunting.
    tail = slice(2 * len(frames) // 3, None)
    settled = dict(
        shutter_us=int(np.median(exposure[tail])),
        gain=round(float(np.median(gain[tail])), 3),
        awb_r=round(float(np.median(red[tail])), 3),
        awb_b=round(float(np.median(blue[tail])), 3),
    )

    converged_at = len(exposure)
    final = np.median(exposure[tail])
    for i, e in enumerate(exposure):
        if abs(e - final) / final < 0.10:
            converged_at = i
            break

    print(f"  {len(frames)} frames")
    print(f"  exposure   {exposure.min():7.0f} .. {exposure.max():7.0f} us "
          f"-> settles at {settled['shutter_us']} us")
    print(f"  gain       {gain.min():7.3f} .. {gain.max():7.3f} "
          f"-> {settled['gain']}")
    print(f"  AWB red    {red.min():7.3f} .. {red.max():7.3f} "
          f"-> {settled['awb_r']}")
    print(f"  AWB blue   {blue.min():7.3f} .. {blue.max():7.3f} "
          f"-> {settled['awb_b']}")
    print(f"  scene      {lux.min():7.0f} .. {lux.max():7.0f} lux")
    print(f"  frame time {duration.min():7.0f} .. {duration.max():7.0f} us "
          f"(requested {1e6/FRAMERATE:.0f})")
    print(f"  AE within 10% of its final value from frame {converged_at} "
          f"(~{converged_at/FRAMERATE:.2f} s of unusable startup)")

    swing = exposure.max() / max(1.0, exposure.min())
    if swing > 1.5:
        print(f"  -> exposure swings {swing:.1f}x while converging; that whole "
              f"window is unreliable for detection.")
    if duration.max() > 1.05e6 / FRAMERATE:
        print(f"  -> frame time exceeds the requested period: the exposure is "
              f"long enough to cap your frame rate.")
    return settled


# --------------------------------------------------------------------------
# Phase 2: measure one configuration
# --------------------------------------------------------------------------
def measure(label: str, **camera_kwargs) -> dict | None:
    cam = Camera(source=0, width=WIDTH, height=HEIGHT, framerate=FRAMERATE,
                 **camera_kwargs)
    det = OrangeDetector(min_area=200, score_threshold=0.15)

    for _ in range(int(FRAMERATE)):  # let the stream start before timing it
        cam.read_frame()

    times, xs, ys, us, vs, y_max, found = [], [], [], [], [], [], 0
    for _ in range(SAMPLES):
        frame = cam.read_frame()
        if frame is None:
            continue
        times.append(frame.t_capture)
        detection = det.detect(frame.bgr)
        if not detection.found:
            continue
        found += 1
        xs.append(detection.x)
        ys.append(detection.y)
        # Chroma at the blob's centre, at chroma resolution.
        cy, cx = detection.y // 2, detection.x // 2
        us.append(float(np.median(frame.u[cy - 3:cy + 4, cx - 3:cx + 4])))
        vs.append(float(np.median(frame.v[cy - 3:cy + 4, cx - 3:cx + 4])))
        y_max.append(int(frame.y.max()))
    dropped = cam.dropped_frames
    cam.release()

    if found < 10:
        print(f"\n{label}: only {found}/{SAMPLES} detections - is the ball in frame?")
        return None

    dt = np.diff(times)
    xs, ys, us, vs = map(np.array, (xs, ys, us, vs))
    # Scatter around the median, not the mean: if the ball drifted a little,
    # this is less distorted than a plain std.
    scatter_px = float(np.sqrt(np.mean((xs - np.median(xs))**2
                                       + (ys - np.median(ys))**2)))
    result = dict(
        fps=1.0 / float(np.median(dt)),
        jitter_ms=float(np.std(dt) * 1e3),
        rate=found / SAMPLES,
        scatter_px=scatter_px,
        drift_px=float(max(xs.max() - xs.min(), ys.max() - ys.min())),
        u_mean=float(us.mean()), u_std=float(us.std()),
        v_mean=float(vs.mean()), v_std=float(vs.std()),
        clipped=int(np.mean(np.array(y_max) >= 255) * 100),
        dropped=dropped,
    )

    print(f"\n{label}")
    print(f"  frame rate        {result['fps']:6.2f} fps   "
          f"(inter-frame jitter {result['jitter_ms']:.2f} ms, {dropped} stale skipped)")
    print(f"  detection rate    {100*result['rate']:6.1f} %")
    print(f"  centroid scatter  {result['scatter_px']:6.2f} px RMS   "
          f"(total drift {result['drift_px']:.0f} px)")
    print(f"  U inside ball     {result['u_mean']:6.1f} +/- {result['u_std']:.2f}")
    print(f"  V inside ball     {result['v_mean']:6.1f} +/- {result['v_std']:.2f}")
    if result["clipped"]:
        print(f"  luma clipped on {result['clipped']}% of frames - overexposed")
    if result["drift_px"] > 15:
        print("  NOTE: the ball moved during the run, so the scatter number is")
        print("        its motion, not measurement noise. Re-run with it still.")
    return result


def main() -> None:
    print("Put the ball on the plate and leave it STILL.\n")
    settled = observe_auto()

    print(f"\nPhase 2: measuring {SAMPLES} frames per configuration ...")
    auto = measure("AUTO (what you run today)")
    locked = measure(
        f"LOCKED (shutter={settled['shutter_us']} us, gain={settled['gain']}, "
        f"awb={settled['awb_r']},{settled['awb_b']})",
        shutter_us=settled["shutter_us"], gain=settled["gain"],
        awb_gains=(settled["awb_r"], settled["awb_b"]),
    )
    if auto is None or locked is None:
        sys.exit(1)

    print("\n" + "=" * 68)
    print(f"{'':<20}{'AUTO':>12}{'LOCKED':>12}   what it means")
    print(f"{'chroma U drift':<20}{auto['u_std']:>12.2f}{locked['u_std']:>12.2f}"
          f"   detect_yuv bound stability")
    print(f"{'chroma V drift':<20}{auto['v_std']:>12.2f}{locked['v_std']:>12.2f}"
          f"   same")
    print(f"{'centroid scatter':<20}{auto['scatter_px']:>12.2f}{locked['scatter_px']:>12.2f}"
          f"   px RMS -> Kalman R")
    print(f"{'frame rate':<20}{auto['fps']:>12.2f}{locked['fps']:>12.2f}   fps")

    blur_mm = BALL_SPEED_MM_S * settled["shutter_us"] * 1e-6
    print(f"\nmotion blur at {BALL_SPEED_MM_S:.0f} mm/s: {blur_mm:.2f} mm "
          f"({settled['shutter_us']} us of exposure)")
    if blur_mm > 3:
        print("  that is a large smear for a centroid; shorten the shutter and")
        print("  raise the gain to compensate, or light the scene better.")

    print("\nPaste into main.py:\n")
    print("camera = Camera(source=camera_source, width=camera_width,")
    print("                height=camera_height, framerate=camera_framerate,")
    print(f"                shutter_us={settled['shutter_us']}, gain={settled['gain']},")
    print(f"                awb_gains=({settled['awb_r']}, {settled['awb_b']}))")
    print()
    print("and, once you have a real mm_per_px from calibrate.py:")
    print(f"    measurement_noise_std = {locked['scatter_px']:.2f} * mm_per_px"
          f"   # = {locked['scatter_px']:.2f} px RMS, measured")
    print("\nRe-run calibrate_yuv.py AFTER locking: the bounds shift with AWB.")


if __name__ == "__main__":
    main()
