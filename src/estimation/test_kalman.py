"""Visual test for the ball-balancing robot's Kalman filter."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Adjust this import if the project's KalmanFilter lives in another module.
from kalman_filter import KalmanFilter  # Assumes your class is in kalman_filter.py

# Simulation/configuration constants requested for this test.
DT = 0.1
g = 9.81
MEASUREMENT_NOISE_VARIANCE = 25.0  # 5-pixel standard deviation in x and y
ACCEL_VARIANCE = 0.1
STEPS = 250

def simulate_bouncing_ball(
    steps: int = STEPS,
    dt: float = DT,
    gravity: float = g,
) -> np.ndarray:
    """Return the true [x, y] trajectory of a ball bouncing at y=0."""
    position = np.array([100.0, 50.0])
    velocity = np.array([2.0, -1.0])
    trajectory = np.empty((steps, 2), dtype=float)

    for index in range(steps):
        trajectory[index] = position

        # x has constant velocity; y has downward/upward acceleration +g.
        position = position + velocity * dt
        position[1] += 0.5 * gravity * dt**2
        velocity[1] += gravity * dt

        # Reflect the state at the floor to produce a bouncing trajectory.
        if position[1] < 0.0:
            position[1] = -position[1]
            velocity[1] = -velocity[1]

    return trajectory

def make_kalman_filter(dt: float = DT) -> KalmanFilter:
    """Build a constant-velocity filter with acceleration process noise."""
    initial_x = 100.0
    initial_y = 50.0
    accel_variance = ACCEL_VARIANCE
    # Optional: override default F, G, R, H if needed
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    G = np.zeros((4, 2))
    G[:2, :] = np.array([[0.5, 0], [0, 0.5]]) * (dt**2)
    G[2:, :] = np.array([[1, 0], [0, 1]]) * dt
    R = np.eye(2) * MEASUREMENT_NOISE_VARIANCE
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    return KalmanFilter(
        initial_x=initial_x,
        initial_y=initial_y,
        accel_variance=accel_variance,
        F=F,
        G=G,
        R=R,
        H=H,
    )

def run_filter(measurements: np.ndarray, dt: float = DT) -> np.ndarray:
    """Run predict/update for each camera measurement and return [x, y]."""
    kalman_filter = make_kalman_filter(dt)
    filtered = np.empty_like(measurements)

    for index, measurement in enumerate(measurements):
        kalman_filter.predict(dt)  # Pass dt to update F and G internally
        updated = kalman_filter.update(measurement[0], measurement[1], MEASUREMENT_NOISE_VARIANCE)
        filtered[index] = kalman_filter._current_state[:2].flatten()

    return filtered

def main() -> None:
    """Simulate measurements, filter them, and display comparison plots."""
    rng = np.random.default_rng(7)
    true_trajectory = simulate_bouncing_ball()
    measurement_noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(MEASUREMENT_NOISE_VARIANCE),
        size=true_trajectory.shape,
    )
    measurements = true_trajectory + measurement_noise
    filtered_trajectory = run_filter(measurements)
    time = np.arange(len(true_trajectory)) * DT

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(true_trajectory[:, 0], true_trajectory[:, 1], label="True trajectory", linewidth=2)
    axes[0].scatter(measurements[:, 0], measurements[:, 1], s=10, alpha=0.35, label="Camera measurements")
    axes[0].plot(filtered_trajectory[:, 0], filtered_trajectory[:, 1], label="Kalman filter", linewidth=2)
    axes[0].set(title="Ball path", xlabel="x (pixels)", ylabel="y (pixels)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, true_trajectory[:, 1], label="True y", linewidth=2)
    axes[1].scatter(time, measurements[:, 1], s=10, alpha=0.35, label="Measured y")
    axes[1].plot(time, filtered_trajectory[:, 1], label="Filtered y", linewidth=2)
    axes[1].set(title="Vertical position over time", xlabel="time (s)", ylabel="y (pixels)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle("Kalman Filter Test — Bouncing Ball")
    plt.show()

if __name__ == "__main__":
    main()