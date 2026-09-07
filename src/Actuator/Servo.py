import numpy as np
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685


class Servo:
    """
    Repère logique centré sur la position neutre :

    -pi     = butée basse
     0      = neutre (plateau plat, grâce à l'offset)
    +pi     = butée haute

    L'offset_rad définit où se trouve le neutre mécanique
    dans le repère brut du servo (0..pi).
    Commande interne = offset + angle_commandé, clampé à [0, pi].
    """

    def __init__(
        self,
        pca,
        channel,
        min_pulse=500,
        max_pulse=2400,
        name=None,
        reverse=False,
        offset_rad=0.0,
    ):
        self.pca = pca
        self.channel = channel
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self.name = name or f"servo_{channel}"
        self.reverse = reverse
        self.offset_rad = offset_rad

        self._angle_rad = 0.0

    def _angle_to_duty_cycle(self, servo_angle_deg):
        servo_angle_deg = np.clip(servo_angle_deg, 0.0, 180.0)

        pulse_us = self.min_pulse + (
            servo_angle_deg / 180.0
        ) * (self.max_pulse - self.min_pulse)

        period_us = 1_000_000 / self.pca.frequency
        duty_cycle = (pulse_us / period_us) * 65535

        return int(np.clip(duty_cycle, 0, 65535))

    @property
    def range_min(self):
        """Angle minimum commandable (rad) avant butée basse."""
        return -self.offset_rad

    @property
    def range_max(self):
        """Angle maximum commandable (rad) avant butée haute."""
        return np.pi - self.offset_rad

    def set_angle(self, angle_rad):
        """
        Reçoit un angle en radians dans le repère centré.
        0 = neutre (plateau plat).
        """
        angle_rad = float(angle_rad)

        # Convertir du repère centré vers le repère brut servo [0, pi]
        raw = self.offset_rad + angle_rad
        raw = np.clip(raw, 0.0, np.pi)

        servo_angle_deg = np.degrees(raw)

        if self.reverse:
            servo_angle_deg = 180.0 - servo_angle_deg

        duty_cycle = self._angle_to_duty_cycle(servo_angle_deg)
        self.pca.channels[self.channel].duty_cycle = duty_cycle

        self._angle_rad = angle_rad

    def get_angle(self):
        """Retourne le dernier angle commandé (repère centré)."""
        return self._angle_rad

    def release(self):
        self.pca.channels[self.channel].duty_cycle = 0


def create_pca(address=0x40, frequency=50):
    i2c = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c, address=address)
    pca.frequency = frequency
    return pca