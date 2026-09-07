from __future__ import annotations

from Actuator.Servo import Servo


class Arm:
    """One leg of the 3-arm parallel platform mechanism.

    Chain: motor axis (radius L3 from center) -> proximal link (L2) ->
    elbow -> distal link (L1) -> spherical joint on the plate.

    Modeled as a planar 2-link mechanism in the vertical plane through the
    base center at this leg's mount angle (theta_i): the motor axis and
    this leg's plate attachment point are assumed to sit at the same
    azimuth around the center, differing only in radius and height.
    """

    def __init__(
        self,
        servo: Servo,
        L1: float,
        L2: float,
        L3: float,
        mount_angle_rad: float,
    ):
        self.servo = servo
        self.L1 = L1  # distal link: elbow -> spherical joint
        self.L2 = L2  # proximal link: motor axis -> elbow
        self.L3 = L3  # base radius: plate center -> motor axis
        self.mount_angle_rad = mount_angle_rad  # theta_i: this leg's azimuth

    def set_angle(self, theta: float):  # in radians
        self.servo.set_angle(theta)

    def get_azimuth(self) -> float:
        """Return this arm's azimuth angle (theta_i) in radians."""
        return self.mount_angle_rad