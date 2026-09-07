"""Inverse kinematics: plate orientation -> per-arm servo angles.

Notation (matches the project's math notes):
    n = (alpha, beta, gamma)   plate normal vector (need not be unit length)
    h                          height of the plate center C = (0, 0, h)
    L                          plate half-width: center -> spherical joint
    theta_i                    arm i's azimuth (0, 120, 240 degrees)
    u_theta = (cos, sin, 0)    horizontal direction of arm i's plane
    P_i                        spherical joint at the top of arm i
    Arm.L3                     base radius: center -> motor axis
    Arm.L2                     proximal link: motor axis -> elbow
    Arm.L1                     distal link: elbow -> spherical joint
    q_i                        commanded joint angle of arm i

Geometry model:
    - P_i's horizontal position is fixed at (L*cos(theta_i), L*sin(theta_i))
      (the plate's rotation is assumed small enough that attachment points
      don't shift horizontally - only their height changes). P_i's height
      is then whatever the plate's plane equation requires there:
          alpha*x + beta*y + gamma*(z - h) = 0
      solved at (x, y) = (L*cos(theta_i), L*sin(theta_i)) for z.
    - Each arm is then solved as a planar 2-link (L2, L1) problem in the
      vertical plane at azimuth theta_i, since the motor axis sits at
      (L3*cos(theta_i), L3*sin(theta_i), 0) - the same azimuth as P_i.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.control.arm import Arm


###VARIABLES
L = 5  # plate half-width: center -> spherical joint (m)

@dataclass
class PlateOrientation:
    n: tuple[float, float, float]  # (alpha, beta, gamma), need not be unit
    h: float  # height of plate center above the motor-axis (z=0) plane

@dataclass
class VectorPosition:
    x:float
    y:float
    z:float


def incline_board(correction): # TODO clarify the correction parameter type
    "Do the sequence of reaction to incline the plate"
    target_orientation = normal_vector_board(pitch=correction.pitch_rad, roll=correction.roll_rad, height=correction.height_m)
    list_position = solve_end_effector_positions(target_orientation=target_orientation, arms=correction.arms, L=L)
    bearing_positions = solve_bearing_positions(target_orientation=target_orientation, arms=correction.arms, arms_end_effector=list_position, L=L)
    arms_angles = solve_servo_angles(correction.arms, bearing_positions)
    return arms_angles
    
def normal_vector_board(pitch: float, roll: float, height: float) -> PlateOrientation:
    """Convert pitch/roll angles to a plate normal vector.

    Pitch and roll are in radians, with the plate's local axes aligned with
    the global axes when both are zero. The plate's center is at (0, 0, h).
    """
    alpha = np.sin(roll)
    beta = -np.sin(pitch)
    gamma = np.sqrt(1 - alpha**2 - beta**2)  # positive z-axis
    return PlateOrientation(n=(alpha, beta, gamma), h=height)

def solve_end_effector_positions(
    target_orientation: PlateOrientation, arms: np.ndarray, L: float) -> np.ndarray:
    """Compute the position needed for the end effector of each arm to have the correct angle for the plate orientation."""
    alpha, beta, gamma = target_orientation.n
    h = target_orientation.h

    positions = np.empty(arms.shape[0], dtype=object)
    for i in range(arms.shape[0]):
        arm = arms[i]
        theta_i = arm.get_azimuth()
        cos_t = np.cos(theta_i)
        sin_t = np.sin(theta_i)

        d_i = alpha * cos_t + beta * sin_t  # u_i . n
        N_i = np.sqrt(d_i**2 + gamma**2)

        x_i = L * gamma * cos_t / N_i
        y_i = L * gamma * sin_t / N_i
        z_i = h - L * d_i / N_i

        positions[i] = VectorPosition(x=x_i, y=y_i, z=z_i)

    return positions

def solve_bearing_positions(target_orientation: PlateOrientation, arms: np.ndarray, arms_end_effector: np.ndarray, L: float) -> np.ndarray:
    """Compute the position of each arm's bearing joint based on the end effector position and the plate orientation."""
    bearing_positions = np.empty(arms.shape[0], dtype=object)
    for i, (arm, end_effector) in enumerate(zip(arms, arms_end_effector)):
        A = ( arm.L3 - end_effector.x )/ end_effector.z
        B = ( end_effector.x**2 + end_effector.y**2 + end_effector.z**2 + arm.L2**2 - arm.L1**2 - arm.L3**2 ) / ( 2 * end_effector.z )
        ## Then we use those for x :
        D = A**2 + 1
        E = 2*(A*B - arm.L3)
        F = arm.L3**2 + B**2 - arm.L2**2
        x_bearing = (-E - np.sqrt(E**2 - 4*D*F)) / (2*D)
        z_bearing = A*x_bearing + B
        y_bearing = end_effector.y * (z_bearing / end_effector.z)
        bearing_position = VectorPosition(x=x_bearing, y=y_bearing, z=z_bearing)
        arm.bearing_position = bearing_position
        bearing_positions[i] = bearing_position
    return bearing_positions


def solve_servo_angles(arms: np.ndarray, bearing_positions: np.ndarray) -> np.ndarray:
    """Compute the servo angles for each arm based on the bearing positions."""
    angles = np.empty(arms.shape[0], dtype=float)
    for i, (arm, bearing_position) in enumerate(zip(arms, bearing_positions)):
        cos_theta = (bearing_position.x - arm.L3) / arm.L2
        sin_theta = bearing_position.z / arm.L2
        angles[i] = np.arctan2(sin_theta, cos_theta)
    return angles