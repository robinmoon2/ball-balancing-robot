import numpy as np
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

class Servo:
    """
    Repère logique :

    0 rad       = bas
    pi / 2 rad  = gauche horizontal
    pi rad      = haut
    """

    def __init__(
        self,
        pca,
        channel,
        min_pulse=500,
        max_pulse=2400,
        name=None,
        reverse=False
    ):
        self.pca = pca
        self.channel = channel
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self.name = name or f"servo_{channel}"
        self.reverse = reverse

        self._angle_rad = 0.0

    def _angle_to_duty_cycle(self, servo_angle_deg):
        # Sécurité : le PWM reste entre 0 et 180 degrés
        servo_angle_deg = np.clip(servo_angle_deg, 0.0, 180.0)

        pulse_us = self.min_pulse + (
            servo_angle_deg / 180.0
        ) * (self.max_pulse - self.min_pulse)

        period_us = 1_000_000 / self.pca.frequency

        duty_cycle = (pulse_us / period_us) * 65535

        return int(np.clip(duty_cycle, 0, 65535))

    def set_angle(self, angle_rad):
        """
        Reçoit un angle en radians.

        0       = bas
        pi / 2  = horizontal gauche
        pi      = haut
        """

        # Limite de sécurité du repère mathématique
        angle_rad = np.clip(
            float(angle_rad),
            0.0,
            np.pi
        )

        # Conversion du repère logique 0..pi vers la commande servo 0..180°
        servo_angle_deg = np.degrees(angle_rad)

        if self.reverse:
            servo_angle_deg = 180.0 - servo_angle_deg

        duty_cycle = self._angle_to_duty_cycle(servo_angle_deg)

        self.pca.channels[self.channel].duty_cycle = duty_cycle
        self._angle_rad = angle_rad

    def get_angle(self):
        return self._angle_rad

    def release(self):
        self.pca.channels[self.channel].duty_cycle = 0
    
def create_pca(address=0x40, frequency=50):
    """Initialise et retourne le PCA9685."""
    i2c = busio.I2C(SCL, SDA)

    pca = PCA9685(i2c, address=address)
    pca.frequency = frequency

    return pca