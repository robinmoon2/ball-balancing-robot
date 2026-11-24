"""
LQR gain computation, used instead of the PID to control the robot.

LQR is a state-feedback controller u = -K x whose gain K minimizes
    J = integral( x^T Q x + u^T R u ) dt
for the linear model x' = A x + B u. Q penalizes state error, R penalizes
control effort.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

from src.utils import StateSpaceModel


class LQR:
    def __init__(self, S: StateSpaceModel, Q: np.ndarray, R: np.ndarray):
        self._S = S
        self._Q = np.asarray(Q, dtype=float)
        self._R = np.asarray(R, dtype=float)
        self._K = self.compute_lqr_gain()

    def get_gain(self) -> np.ndarray:
        return self._K

    def set_gain(self, K: np.ndarray) -> None:
        self._K = K

    def compute_lqr_gain(self) -> np.ndarray:
        """Solve the continuous-time algebraic Riccati equation (CARE) for P,
        then K = R^-1 B^T P."""
        P = solve_continuous_are(self._S.A, self._S.B, self._Q, self._R)
        return np.linalg.solve(self._R, self._S.B.T @ P)
