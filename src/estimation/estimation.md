# Estimator — Specification

## 1. Purpose

Provide a clean, time-consistent estimate of the ball's state
(position and velocity) in the plate frame, from noisy and possibly
intermittent detections.

## 2. Non-goals

- Does NOT convert pixels to millimeters (done upstream).
- Does NOT predict future positions for actuator-lag compensation
  (separate concern).
- Does NOT decide whether the ball is "lost" for safety purposes
  (it only reports validity; the controller decides what to do).

## 3. Inputs

- `measurement`: optional 2D position
    - type: `tuple[float, float] | None`
    - frame: plate-centered, +X right, +Y up
    - units: mm
    - `None` means "no detection this tick"
- `t`: timestamp of the measurement
    - type: `float`, seconds (monotonic clock)
    - source: capture time of the camera frame, NOT loop time

## 4. Outputs

- `BallEstimate` dataclass with:
    - `x, y`: mm, plate frame
    - `vx, vy`: mm/s, plate frame
    - `t`: timestamp the estimate refers to
    - `valid`: bool — false during warm-up or after long dropout
    - (optional) `cov`: 4x4 covariance for diagnostics

## 5. Internal state

- Last estimate (mean + covariance)
- Last timestamp
- Consecutive-miss counter

## 6. Behavior

- **First call:** initialize state from the first measurement,
  velocity = 0, mark `valid=False` for the first N ticks.
- **Normal call (measurement present):** predict forward by dt,
  then update with measurement.
- **Missing measurement:** predict forward only, do NOT update.
- **Long dropout (> T_lost seconds):** mark `valid=False`, but
  keep predicting (controller will decide to disengage).
- **Negative or zero dt:** ignore the call, log a warning.

## 7. Assumptions

- X and Y dynamics are decoupled → two independent 1D filters.
- Constant-velocity motion model with process noise absorbing
  acceleration from plate tilt.
- Measurement noise is Gaussian, ~roughly isotropic, std ≈ 1–2 mm
  (to be characterized empirically).

## 8. Tuning parameters (configurable)

- `process_noise_std`: mm/s² equivalent
- `measurement_noise_std`: mm
- `warmup_ticks`: int
- `max_dropout_seconds`: float

## 9. Failure modes & how they're handled

| Mode | Detection | Response |
|---|---|---|
| First call ever | internal flag | initialize, return invalid |
| Single-frame dropout | measurement is None | predict only |
| Long dropout | counter > threshold | predict + invalid flag |
| Time going backward | dt <= 0 | skip, warn |
| NaN in measurement | isnan check | treat as dropout |
