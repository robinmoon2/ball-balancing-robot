import numpy as np


class KalmanFilter:
    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        accel_variance: float,
        F: np.ndarray = np.array([[1, 0], [0, 1]]),
        G: np.ndarray = np.array([[0.5, 0], [0, 0.5]]),
        R: np.ndarray = np.array([[1, 0], [0, 1]]),
        H: np.ndarray = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
        g: float = 9.81,
    ):

        # State transition matrix
        self.G = G  # Control input matrix
        self.H = H
        self.R = R  # Measurement noise covariance
        # Base G for velocity terms (1 factor)
        self.G_vel = np.array([[1, 0], [0, 1]])
        # Base G for position terms (0.5 factor)
        self.G_pos = np.array([[0.5, 0], [0, 0.5]])
        self._accel_variance = accel_variance
        self._current_state = np.array(
            [[initial_x], [initial_y], [0], [0]]
        )  # Initial state
        self.P = np.diag([4.0, 4.0, 90000.0, 90000.0])  # Initial estimate covariance
        self.g = g  # Gravity acceleration in m/s²

    def predict(self, dt: float):
        # Predict the next state based on the current state and control input
        # x = F *x
        # P = F P Ft + G a Gt
        self.F = np.array(
            [
                [1, 0, dt, 0],  # x = x + vx*dt
                [0, 1, 0, dt],  # y = y + vy*dt
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy + g*dt (but g is in Q)
            ]
        )

        self.G = np.zeros((4, 2))
        self.G[:2, :] = self.G_pos * (dt**2)  # Position terms (dt²)
        self.G[2:, :] = self.G_vel * dt  # Velocity terms (dt)

        # Control input: gravity in y-direction
        a = np.array([[0], [self.g]])
        new_state = self.F.dot(self._current_state) + self.G.dot(a)  # State prediction
        Q = self.G.dot(self.G.T) * self._accel_variance  # Process noise covariance
        new_P = self.F.dot(self.P).dot(self.F.T) + Q  # Covariance prediction

        self._current_state = new_state
        self.P = new_P

    def update(self, x_meas: np.ndarray, y_meas: np.ndarray):
        # y = z - Hx
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + Ky
        # P = (I - KH)P

        z = np.array([[x_meas], [y_meas]])  # Measurement vector

        y = z - self.H.dot(self._current_state)  # Measurement residual
        S = self.H.dot(self.P).dot(self.H.T) + self.R  # Residual covariance

        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))  # Kalman gain

        new_state = self._current_state + K.dot(y)  # Updated state estimate
        I_KH = np.eye(4) - K.dot(self.H)  # Identity minus Kalman gain times H
        new_P = I_KH.dot(self.P).dot(I_KH.T) + K.dot(self.R).dot(
            K.T
        )  # Updated estimate covariance

        self.P = new_P
        self._current_state = new_state

    def get_current_state(self) -> np.ndarray:
        return self._current_state
