"""Entry point: connects PlateSim <-> PlateController."""

from __future__ import annotations
import math
import time

from src.control.pid import PID
from src.control.controller import PlateController
from src.simulation.plate_sim import PlateSim, SimParams
from src.utils import BallEstimate
from src.simulation.renderer import Renderer


def main(realtime: bool = True, duration: float = 30.0) -> None:
    dt = 1.0 / 200.0  # physics at 200 Hz
    control_every = 3  # control at ~66 Hz (every 3 physics steps)

    params = SimParams(
        plate_radius=100.0,
        friction=0.8,
        actuator_lag_tau=0.04,
        measurement_noise_std=0.8,
        detection_dropout_prob=0.0,
    )
    sim = PlateSim(params=params, seed=42)
    sim.reset(x=60.0, y=-40.0)

    # Starting gains — you will TUNE these. These are a reasonable starting point.
    pid_x = PID(
        kp=0.0015,
        ki=0.0,
        kd=0.0010,
        output_limits=(-math.radians(15), math.radians(15)),
        integral_limits=(-0.5, 0.5),
        derivative_filter=0.85,
    )
    pid_y = PID(
        kp=0.0015,
        ki=0.0,
        kd=0.0010,
        output_limits=(-math.radians(15), math.radians(15)),
        integral_limits=(-0.5, 0.5),
        derivative_filter=0.85,
    )

    controller = PlateController(pid_x, pid_y, max_tilt_rad=math.radians(15))
    controller.set_target(0.0, 0.0)

    renderer = Renderer(plate_radius=params.plate_radius)

    t = 0.0
    step_i = 0
    last_meas = (sim.s.x, sim.s.y, True)
    # Simple velocity estimate via finite-difference (we'll make this proper later)
    prev_xy: tuple[float, float] | None = None
    vx = vy = 0.0
    control_dt = dt * control_every

    t_wall_start = time.time()
    while t < duration and sim.s.on_plate:
        # Physics step
        sim.step(dt)

        # Control step (slower than physics)
        if step_i % control_every == 0:
            xm, ym, found = sim.measure()
            if found and prev_xy is not None:
                vx = (xm - prev_xy[0]) / control_dt
                vy = (ym - prev_xy[1]) / control_dt
            if found:
                prev_xy = (xm, ym)
            state = BallEstimate(x=xm, y=ym, vx=vx, vy=vy, t=t, valid=found)
            cmd = controller.update(state, control_dt)
            sim.set_command(cmd.roll, cmd.pitch)

        # Render every ~30 ms
        if step_i % 6 == 0:
            renderer.update(t, sim.s.x, sim.s.y)

        t += dt
        step_i += 1
        if realtime:
            target_wall = t_wall_start + t
            sleep = target_wall - time.time()
            if sleep > 0:
                time.sleep(sleep)

    print(f"Done. on_plate={sim.s.on_plate}, final=({sim.s.x:.1f},{sim.s.y:.1f})")
    import matplotlib.pyplot as plt

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
