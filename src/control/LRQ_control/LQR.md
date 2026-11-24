# LQR controller

Code: `src/control/controller_lqr.py` (controller), `LQR.py` (gain),
`StateSpaceModel` in `src/utils.py` (model).

## 1. States and input

State vector (plate frame, origin = centre of the camera image):

| State | Meaning               | Unit |
|-------|-----------------------|------|
| `x`   | ball position error X | mm   |
| `y`   | ball position error Y | mm   |
| `vx`  | ball velocity X       | mm/s |
| `vy`  | ball velocity Y       | mm/s |

The positions are errors, measured from the target: `x = ball_x - target_x`.
All four come from the Kalman filter (`BallEstimate`).

Input vector: `u = [roll, pitch]` in rad. These are the plate angles sent to
the inverse kinematics.

## 2. Model

For small angles, a ball rolling on a tilted plate accelerates in proportion
to the tilt:

```
ax = +g_eff * pitch        (+pitch -> ball rolls toward +x)
ay = -g_eff * roll         (+roll  -> ball rolls toward -y)
```

Here `g_eff` = rolling factor × g, in mm/s² per rad:

- solid ball: 5/7 × 9810 ≈ 7000 (`G_EFF` in `.env`)
- hollow ball: 3/5 × 9810 ≈ 5900

In state-space form, `ẋ = A·x + B·u`:

```
      x  y  vx vy              roll    pitch
A = [ 0  0  1  0 ]       B = [   0       0    ]
    [ 0  0  0  1 ]           [   0       0    ]
    [ 0  0  0  0 ]           [   0     g_eff  ]
    [ 0  0  0  0 ]           [ -g_eff    0    ]
```

The model is two independent double integrators, one for X driven by pitch
and one for Y driven by roll. Friction, servo dynamics and delay are not
modelled.

## 3. How LQR works

LQR finds the state-feedback law

```
u = -K · x
```

that minimises the cost

```
J = ∫ ( xᵀ Q x + uᵀ R u ) dt
```

- `Q` (4×4) penalises the ball being off-centre and moving fast.
- `R` (2×2) penalises tilting the plate.

Large `Q` relative to `R` gives an aggressive controller. Large `R` gives a
gentle one.

`K` is computed once at startup:

1. Solve the continuous algebraic Riccati equation for `P`:
   `Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0`
   using `scipy.linalg.solve_continuous_are`.
2. Compute `K = R⁻¹ Bᵀ P`, a 2×4 matrix.

At every tick the controller computes `[roll, pitch] = -K · [x, y, vx, vy]`.
The base `Controller` class then clamps the tilt magnitude to `MAX_TILT_DEG`.

## 4. Choosing Q and R (Bryson's rule)

`ControllerLQR.from_bryson` sets each weight to 1 / (largest acceptable value)²:

```
Q = diag(1/x_max², 1/x_max², 1/v_max², 1/v_max²)
R = diag(1/u_max², 1/u_max²)
```

| Symbol  | `.env` variable    | Meaning                     |
|---------|--------------------|-----------------------------|
| `x_max` | `LQR_MAX_POS_MM`   | acceptable position error   |
| `v_max` | `LQR_MAX_VEL_MM_S` | acceptable ball speed       |
| `u_max` | `MAX_TILT_DEG`     | acceptable tilt (in rad)    |

For this model, the resulting gain on each axis has a closed form:

```
K_pos = u_max / x_max
K_vel = sqrt( u_max² / v_max²  +  2·K_pos / g_eff )
```

The closed loop then behaves like a mass-spring-damper:

```
ωn = sqrt(g_eff · K_pos)          (speed of response)
ζ  = g_eff · K_vel / (2·ωn)       (damping)
```

Example: `g_eff=7000`, `u_max=20°`, `x_max=150`, `v_max=300`
→ `K_pos=0.00233`, `K_vel=0.00142`, `ωn≈4.0 rad/s`, `ζ≈1.24`.

The two gains also appear as entries of K. Roll is driven by y and pitch by x,
so the other entries are 0:

```
K = [ 0       -K_pos    0       -K_vel ]   -> roll
    [ K_pos    0        K_vel    0     ]   -> pitch
```

**Tuning**
- Lower `LQR_MAX_POS_MM` → stiffer centring.
- Lower `LQR_MAX_VEL_MM_S` → more damping.

## 5. Limitations

- **No integral term.** If the "flat" pose is not really level, the ball
  settles off-centre, at a distance of offset_angle / K_pos. Level the plate
  first.
- **No delay in the model.** The real loop has about 100–200 ms of camera and
  servo latency. If the gains are too stiff the ball oscillates, so keep `ωn`
  around 4 rad/s or lower.
- **Small-angle model.** It is accurate up to about 15–20° of tilt.
