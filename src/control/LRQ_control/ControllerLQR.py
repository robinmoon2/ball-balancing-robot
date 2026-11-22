import numpy as np 
from src.utils import StateSpaceModel, State, BallEstimate, PlateCommand
from src.control.LRQ_control.LQR import LQR


class ControllerLQR:
    def __init__(self, S: StateSpaceModel, Q: np.ndarray, R: np.ndarray):
        self.lqr = LQR(S, Q, R)
        self.K = self.lqr.get_gain()
        self.target = (0.0, 0.0) # mm
        self._last_command = PlateCommand(roll=0.0, pitch=0.0) # roll, pitch

    def compute_control(self, state: State) -> np.ndarray:
        """
        Compute the control input using the LQR gain matrix K and the current state.
        """
        # u = -K * x
        u = -self.K @ state.to_array()
        return u

    def set_target(self, target_state: State):
        """
        Set the target state for the controller. The control input will be computed
        based on the difference between the current state and the target state.
        """
        self.target_state = target_state
    