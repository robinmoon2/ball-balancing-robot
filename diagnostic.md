# Ball-Balancing Robot — Lag Diagnostic

_Date: 2026-10-05 · Raspberry Pi 4B (4 GB) · Camera Module 3 (IMX708) · PCA9685 @ 0x40_

## TL;DR

**The lag is mostly in the software, not the hardware.** The Pi, camera and I2C bus are all healthy.
Three software problems explain most of the "laggy" / twitchy behaviour:

1. **The derivative gain has the wrong sign** (`kd=-0.005`). It pushes the ball instead of braking it, which causes oscillation.
2. **The D term is mostly noise.** With the ball sitting still, it alone commands ±0.107 rad of jitter. The tilt limit is 0.140 rad.
3. **The colour detector uses 24 ms of the 33 ms frame budget.** A ready-made detector that takes 0.56 ms (`detect_yuv`) is in the code but switched off.

There is also one **urgent non-lag problem**: the **git repository is corrupted** (`fatal: bad object HEAD`) after an unclean shutdown. Back up your work first.

---

## Priority table

| # | Problem | Layer | Impact on lag/stability | Effort | Priority |
|---|---------|-------|-------------------------|--------|----------|
| 1 | `kd` is negative, so the D term is anti-damping | Controller | 🔴 Very high (oscillation) | 1 line | **P0** |
| 2 | D term differentiates noisy position (±0.107 rad jitter) | Controller | 🔴 Very high (twitching) | ~5 lines | **P0** |
| 3 | Git repo corrupted / SD card had an unclean shutdown | System | 🔴 Risk of losing code | 10 min | **P0** |
| 4 | BGR float detector takes 24 ms per frame | Vision | 🟠 High (+24 ms, blocks higher FPS) | Calibrate + 1 flag | **P1** |
| 5 | Software slew limit of 2 rad/s caps servo speed | Actuation | 🟠 High (slow response) | 1 line | **P1** |
| 6 | Auto exposure and continuous autofocus are on | Camera | 🟠 Medium–high (blur, latency, detection drift) | Flags exist | **P1** |
| 7 | Kalman process noise too low (200 vs about 1500) | Estimation | 🟡 Medium (5.7 mm lag during acceleration) | 1 line | **P2** |
| 8 | 30 FPS while the sensor can do 120 FPS | Camera | 🟡 Medium (after #4) | 1 line | **P2** |
| 9 | 3 `print()` calls every tick in `Actuation._write` | Actuation | 🟡 Low–medium (can block over SSH) | Delete | **P2** |
| 10 | No latency compensation (acts on a ~100 ms old position) | Estimation/Control | 🟡 Medium | ~10 lines | **P3** |
| 11 | I2C at 100 kHz | Hardware config | 🟢 Low (~1 ms per tick) | 1 config line | **P3** |
| 12 | Servo PWM at 50 Hz | Hardware | 🟢 Low–medium (0–20 ms) | Depends on servo model | **P3** |
| 13 | Servo power board has no bulk capacitor | Hardware | ❓ Unknown (could cause brown-outs/jerks) | Solder a cap | **Check** |
| 14 | `mm_per_px=0.14` is an uncalibrated guess | Vision | 🟢 Gain scaling only | Run calibrate | **P3** |

---

## What was measured

### Hardware health: ✅ OK

| Check | Result | Verdict |
|---|---|---|
| `vcgencmd get_throttled` | `0x0` (no under-voltage or throttling since boot) | ✅ |
| CPU temperature | 56 °C, 1.8 GHz | ✅ |
| RAM / swap | 2.4 GB free, swap unused | ✅ |
| CPU load (idle) | 0.12 | ✅ |
| Camera frame interval | mean 33.4 ms, p95 36.8 ms, max 40.1 ms at 30 FPS | ✅ steady |
| I2C: PCA9685 transaction | 0.51 ms each (bus at 97.5 kHz) | ✅ (~1.5 ms/tick for 3 servos) |
| I2C devices | `0x40` PCA9685, `0x70` (PCA9685 all-call) | ✅ |
| Kernel log at boot | `EXT4-fs: orphan cleanup on readonly fs` | ⚠️ **last shutdown was unclean** |
| SD card | generic "ASTC" 28 GB | ⚠️ budget card, see #3 |

> Caveat: `get_throttled=0x0` only covers **this boot** (33 min, robot not running). Re-check it **right after a balancing run** with servos moving. A non-zero value there means the power supply is too weak.

### Software timing

| Stage | Measured cost | Notes |
|---|---|---|
| `OrangeDetector.detect()` (BGR, float32) | **23.9 ms** | 23.1 ms of that is `_orangeness()` |
| `OrangeDetector.detect_yuv()` | **0.56 ms** | ~40× faster, needs calibration |
| I420→BGR conversion | 0.64 ms | |
| Inverse kinematics | 0.12 ms | negligible |
| I2C writes (3 servos) | ~1.5 ms | |
| Main loop (from `tuning_log.csv`) | 35.5 ms mean, ~28 Hz | camera-bound, 2 ticks >100 ms |

> ⚠️ In the only run in `tuning_log.csv`, the ball **was never detected** (`found=0` on all 550 ticks). So there is no closed-loop data yet. Record a run with the ball on the plate after the fixes.

### Estimated end-to-end latency (ball moves → plate reacts)

| Step | Time |
|---|---|
| Exposure (auto, up to a full frame) | ~15–33 ms |
| Sensor readout + ISP + pipe | ~33 ms (est.) |
| Detection (BGR) | 24 ms |
| Kalman smoothing lag (q=200) | equivalent to several ms |
| I2C + prints | ~2 ms+ |
| PWM refresh at 50 Hz | 0–20 ms |
| Slew limit 2 rad/s + servo mechanics | ~50–100 ms (est.) |
| **Total** | **≈ 130–210 ms** |

Rule of thumb: a ball-balancing plate feels responsive below about 60–80 ms. The fixes below can roughly halve the total.

---

## Details and fixes

### P0-1 — Derivative gain has the wrong sign 🔴 
`main.py:53-54`
```python
PID_X = PID(kp=0.003, ki=0.001, kd=-0.005)
```
For a PD loop, `kp` and `kd` must have the **same sign**, whatever the plate's sign convention. Both terms act on the same error: the D term must oppose the ball's velocity. With `kd<0` it pushes the ball **faster** in the direction it is already rolling. That gives overshoot, oscillation and a robot that feels late or wobbly.

**Fix:** use `kd = +0.001 … +0.002` to start, then tune it upward.

### P0-2 — D term is noise-dominated | CLEAR
`src/utils.py:281` computes `(error - prev_error) / dt` on the filtered position. Measured with a **still** ball (1.5 mm detector noise, 30 Hz):
- finite-difference derivative noise: **21.4 mm/s std**, so `|kd|·21.4 = 0.107 rad` of random tilt (the limit is 0.140 rad)
- the Kalman filter's own velocity `vx`: **6.6 mm/s std**, about 3× cleaner and with no extra delay

The Kalman filter already estimates velocity, but the controller ignores it.

**Fix (in `Controller.update`):** compute D from `state.vx` / `state.vy` instead of differentiating:
```python
pitch = -(kp*ex + ki*Ix + kd*state.vx)
```
Alternatively, set `derivative_filter=0.5–0.7`, which is simpler but adds lag.


### P1-4 — Slow colour detector 🟠
`src/perception/perception.py:689` defaults to `use_yuv=False`, so each frame takes 24 ms of float math on 640×480×3 pixels.

**Fix:**
1. `python -m src.perception.calibrate_yuv` with the ball on the plate, under the real lighting.
2. Pass `u_max`/`v_min` to `OrangeDetector(...)` and `use_yuv=True` to `Perception(...)` in `main.py`.

Gain: −23 ms per frame. It also frees enough CPU budget for 60–120 FPS (see P2-8).

### P1-5 — Software slew limit throttles the servos 🟠
`src/Actuator/actuation.py:60` sets `max_rate_rad_s=2.0`, and `main.py:107` keeps the default. That is about 115 °/s. Typical hobby servos (MG996R/DS3218) do about 300–500 °/s. The software is making the servos 3–4× slower than they can physically move.

**Fix:** try `Actuation(..., max_rate_rad_s=5.0)` or `None`. The tilt clamp in the controller still protects the mechanics.

### P1-6 — Camera auto-exposure and autofocus 🟠
`main.py:99` creates `Camera(...)` without `shutter_us`/`gain`/`awb_gains`. On the IMX708 (Camera Module 3), `rpicam-vid` also runs **continuous autofocus** by default. Effects:
- Auto exposure in indoor light picks long shutters (~20–33 ms), which causes **motion blur** on a fast ball and adds latency.
- AE/AWB keep re-adjusting as the plate tilts, so the colours the detector thresholds on drift.
- AF can hunt when the plate moves, which blurs frames.

**Fix:**
1. `python -m src.perception.calibrate_exposure`, then pass e.g. `shutter_us=4000, gain=4.0, awb_gains=(…)` to `Camera`.
2. Add `"--autofocus-mode", "manual", "--lens-position", "<dioptres>"` to the `argv` in `src/perception/camera.py:553`. Lens position is 1/distance in metres, e.g. `5.0` for 20 cm.

Brighter lighting on the plate lets you use a short shutter without noise.

### P2-7 — Kalman filter too sluggish 🟡
`main.py:47` sets `process_noise_std = 200.0`, while `estimation.py:38` documents about 1500 as the physically right value (8° tilt ≈ 1000–1800 mm/s²). Simulated with a ball accelerating at 1000 mm/s²:
- q=200: estimate **5.7 mm behind**
- q=1500: 0.5 mm behind

**Fix:** set `process_noise_std = 1000–1500`. If the output is too noisy, filter the D term rather than the position.

### P2-8 — Frame rate 🟡
The sensor offers `1536x864 @ 120 fps`. Once detection is fast (P1-4), use `camera_framerate = 60` (or 90/120) with 640×480. Each step cuts exposure and pipeline delay and gives the Kalman filter more data. Lock a short shutter first (P1-6); at 120 FPS the maximum exposure is 8 ms.

### P2-9 — Console prints in the hot path 🟡
`src/Actuator/actuation.py:221-224` prints 3 lines on **every** tick (~90 lines/s). Over SSH or a slow terminal, `print` can block for milliseconds. `actuation.py:173` also builds a `logger.info(f"...")` string every tick, even when logging is off.

**Fix:** delete the print (the CSV already records the deltas) and use `logger.debug("target pose: %s", target)`.

### P3-10 — No latency compensation 🟡
The estimate is timestamped at frame arrival (good), but the command acts on a position that is ~60–100 ms old. **Fix:** predict forward before control: `x_now = x + vx * (time.monotonic() - state.t + actuator_delay)`. This is cheap and makes a big difference once the other fixes are in.

### P3-11 — I2C bus speed 🟢
The bus runs at 97.5 kHz. Add `dtparam=i2c_arm_baudrate=400000` to `/boot/firmware/config.txt` and reboot. This saves ~1 ms per tick (PCA9685 supports 1 MHz).

### P3-12 — Servo PWM frequency 🟢
`create_pca(frequency=50)` gives a new pulse every 20 ms, so up to 20 ms extra delay. **Only for digital servos** that support it, raise it to 100–200 Hz. `_angle_to_duty_cycle` already uses `pca.frequency`. ⚠️ Analog servos can overheat or buzz above 50 Hz. Check the model first.

### Check-13 — Servo power supply ❓ (couldn't measure from software)
The `servo_power` PCB contains only the PCA9685, a 5 V input and GND, with **no bulk capacitor**. Three servos accelerating together can draw 3–6 A peaks. Voltage sag makes servos slow or jerky and can reset the PCA9685. Manual checks:
- [ ] Servo supply rated **≥ 5 A** at 5–6 V, and **separate** from the Pi's supply (common GND only)
- [ ] Add **470–1000 µF** electrolytic across V+/GND close to the servo connectors
- [ ] Measure V+ with a multimeter while the plate moves fast: it should stay > 4.8 V
- [ ] Run `vcgencmd get_throttled` right after a run: it should read `0x0`
- [ ] Check the mechanics: wiggle the plate by hand with the servos powered. Any free play (servo horn screws, ball joints) shows up as delay and limit cycles.

### P3-14 — Calibration 🟢
`mm_per_px=0.14` is an estimate (`main.py:98`). It only scales your gains, but run `python -m src.perception.calibrate` so the PID numbers mean something physical.

---

## Suggested order

1. **Back up and fix git** (P0-3).
2. Fix the `kd` sign and use the Kalman `vx` for D (P0-1, P0-2). **Biggest improvement for the least work.**
3. Remove the per-tick prints and raise the slew limit (P2-9, P1-5).
4. Calibrate the YUV detector, lock exposure/focus, then go to 60 FPS (P1-4, P1-6, P2-8).
5. Set `process_noise_std≈1000` and add latency prediction (P2-7, P3-10).
6. Check the servo power hardware (Check-13), then make the small config tweaks (P3-11, P3-12).
7. Record a `tuning_log.csv` **with the ball detected** and compare `dt`, oscillation period and settling time before and after.
