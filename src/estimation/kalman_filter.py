import numpy as np
from src.utils import State


class KalmanFilter:
    """Constant-velocity Kalman filter for a ball's (x, y) position.

    No gravity/acceleration term: the ball moves on a plate, not in free
    fall, so acceleration comes from plate tilt rather than a constant
    world-frame force. That acceleration isn't modeled explicitly here —
    it's absorbed into the process noise via accel_variance, per the
    constant-velocity assumption in estimation.md.
    """

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        accel_variance: float,
        R: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ):
        # Measurement matrix and measurement noise covariance
        self.H = (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            if H is None else np.asarray(H, dtype=float)
        )
        self.R = np.eye(2, dtype=float) if R is None else np.asarray(R, dtype=float)

        # Base G for velocity terms (1 factor)
        self.G_vel = np.array([[1, 0], [0, 1]], dtype=float)

        # Base G for position terms (0.5 factor)
        self.G_pos = np.array([[0.5, 0], [0, 0.5]], dtype=float)

        self._accel_variance = accel_variance
        self._current_state = State(
            x=initial_x,
            y=initial_y,
            vx=0.0,
            vy=0.0,
        )

        # Initial estimate covariance
        self.P = np.diag([4.0, 4.0, 90000.0, 90000.0])

        # Innovation (z - Hx, before correction) of the last update(), for
        # logging: should be zero-mean white noise if the filter is tuned.
        self.last_innovation = np.zeros(2)

    @staticmethod
    def _state_to_vector(state: State) -> np.ndarray:
        """Convert State to a 4x1 column vector for matrix calculations."""
        return np.array([[state.x], [state.y], [state.vx], [state.vy]], dtype=float)

    @staticmethod
    def _vector_to_state(vector: np.ndarray) -> State:
        """Convert a 4x1 vector back into a State."""
        vector = np.asarray(vector).reshape(4)
        return State(
            x=float(vector[0]),
            y=float(vector[1]),
            vx=float(vector[2]),
            vy=float(vector[3]),
        )

    def predict(self, dt: float) -> None:
        # Predict the next state based on the current state (constant velocity)
        # x = F x
        # P = F P Fᵀ + Q
        self.F = np.array(
            [
                [1, 0, dt, 0],  # x = x + vx*dt
                [0, 1, 0, dt],  # y = y + vy*dt
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy
            ],
            dtype=float,
        )

        # Map acceleration noise into position and velocity
        G = np.zeros((4, 2), dtype=float)
        G[:2, :] = self.G_pos * (dt**2)  # Position terms (dt²)
        G[2:, :] = self.G_vel * dt  # Velocity terms (dt)

        # State.dot(F) computes F @ state_vector and returns a State
        new_state = self._current_state.dot(self.F)

        # Process noise covariance and covariance prediction
        Q = G @ G.T * self._accel_variance
        new_P = self.F @ self.P @ self.F.T + Q

        self._current_state = new_state
        self.P = new_P

    def update(self, x_meas: float, y_meas: float) -> None:
        # Kalman update:
        # residual = z - Hx
        # S = HPHᵀ + R
        # K = PHᵀS⁻¹
        # x = x + K * residual

        z = np.array([[x_meas], [y_meas]], dtype=float)
        state_vector = self._state_to_vector(self._current_state)

        residual = z - self.H @ state_vector
        self.last_innovation = residual.flatten()
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        updated_vector = state_vector + K @ residual

        # Joseph form for the updated covariance; this is more numerically
        # stable than the simpler form (I - KH)P.
        I_KH = np.eye(4) - K @ self.H
        updated_P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self._current_state = self._vector_to_state(updated_vector)
        self.P = updated_P

    def get_current_state(self) -> State:
        # Return a copy of the current state
        return State(
            x=self._current_state.x,
            y=self._current_state.y,
            vx=self._current_state.vx,
            vy=self._current_state.vy,
        )