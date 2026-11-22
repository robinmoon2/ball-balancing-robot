"""
LQR control algorithm use instead of the PID for the control of the robot. The LQR controller is a state feedback controller that minimizes a cost function to achieve optimal control performance. It is particularly useful for systems with multiple inputs and outputs, such as robotic systems, where it can provide better stability and performance compared to traditional PID controllers.
"""
import numpy as np
from src.utils import StateSpaceModel, State
from scipy.linalg import solve_continuous_are

class LQR:
    def __init__(self, S: StateSpaceModel, Q: np.ndarray, R: np.ndarray):
        self._S = S
        self._Q = Q
        self._R = R
        self._K = self.compute_lqr_gain()
    
    
    # Getter
    def get_gain(self) -> np.ndarray:
        return self._K
    
    # Setter
    def set_gain(self, K: np.ndarray) -> None:
        self._K = K
    
    def compute_lqr_gain(self, states: State) -> np.ndarray:
        """
        Compute the LQR gain matrix K using the state-space model, Q, and R matrices.
        This function uses the continuous-time algebraic Riccati equation (CARE) to compute K.
        """
        # Solve the continuous-time algebraic Riccati equation (CARE)
        P = np.linalg.solve_continuous_are(self._S.A, self._S.B, self._Q, self._R)
        
        # Compute the LQR gain matrix K
        K = np.linalg.inv(self._R) @ self._S.B.T @ P
        
        return K
    