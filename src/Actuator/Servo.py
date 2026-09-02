#!/usr/bin/env python3
"""
Classe Servo pilotant directement le PCA9685, sans dépendance à adafruit_motor.
"""

import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685


class Servo:
    """Représente un servomoteur branché sur un canal du PCA9685."""

    def __init__(self, pca, channel, min_pulse=500, max_pulse=2400,
                 actuation_range=180, name=None, offset=0):
        """
        Args:
            pca (PCA9685): instance partagée du PCA9685.
            channel (int): numéro du canal (0-15).
            min_pulse (int): largeur d'impulsion mini en µs (0°).
            max_pulse (int): largeur d'impulsion maxi en µs (angle max).
            actuation_range (int): plage angulaire du servo.
            name (str): nom du bras (ex: "bras_1").
            offset (float): correction mécanique en degrés.
        """
        self.pca = pca
        self.channel = channel
        self.name = name or f"servo_{channel}"
        self.offset = offset
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self.actuation_range = actuation_range
        self._angle = None

    def _angle_to_duty_cycle(self, angle):
        """
        Convertit un angle (0 -> actuation_range) en duty cycle 16 bits
        pour le PCA9685, en fonction de sa fréquence configurée.
        """
        # Largeur d'impulsion en µs, interpolée entre min_pulse et max_pulse
        pulse_us = self.min_pulse + (angle / self.actuation_range) * (
            self.max_pulse - self.min_pulse
        )

        # Période en µs selon la fréquence PWM du PCA9685 (ex: 50Hz -> 20000µs)
        period_us = 1_000_000 / self.pca.frequency

        # Conversion en valeur 16 bits (0-65535) pour le registre duty_cycle
        duty_cycle = int((pulse_us / period_us) * 65535)
        return max(0, min(65535, duty_cycle))

    def set_angle(self, angle):
        """Positionne le servo à un angle donné (en degrés, avant offset)."""
        real_angle = angle + self.offset
        real_angle = max(0, min(self.actuation_range, real_angle))

        duty_cycle = self._angle_to_duty_cycle(real_angle)
        self.pca.channels[self.channel].duty_cycle = duty_cycle

        self._angle = angle  # angle "logique" demandé

    def get_angle(self):
        return self._angle

    def center(self):
        mid = self.actuation_range / 2
        self.set_angle(mid)

    def release(self):
        """Coupe le signal PWM (le servo devient libre)."""
        self.pca.channels[self.channel].duty_cycle = 0

    def __repr__(self):
        return f"<Servo {self.name} ch={self.channel} angle={self._angle}>"


def create_pca(address=0x40, frequency=50):
    """Crée et retourne une instance PCA9685 configurée."""
    i2c = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c, address=address)
    pca.frequency = frequency
    return pca


# ------------------------------------------------------------------
# Test rapide
# ------------------------------------------------------------------
if __name__ == "__main__":
    pca = create_pca()

    bras_1 = Servo(pca, channel=0, name="bras_1")
    bras_2 = Servo(pca, channel=1, name="bras_2")
    bras_3 = Servo(pca, channel=2, name="bras_3")

    tous_les_bras = [bras_1, bras_2, bras_3]

    try:
        print("Centrage des 3 bras.")
        for bras in tous_les_bras:
            bras.center()
        time.sleep(1)

        print("Test de balayage synchronisé.")
        while True:
            for angle in range(0, 180, 2):
                for bras in tous_les_bras:
                    bras.set_angle(angle)
                time.sleep(0.02)
            time.sleep(0.3)
            for angle in range(180, 0, -2):
                for bras in tous_les_bras:
                    bras.set_angle(angle)
                time.sleep(0.02)
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nCtrl+C détecté. Centrage puis arrêt propre.")
        for bras in tous_les_bras:
            bras.center()
        time.sleep(0.5)
        for bras in tous_les_bras:
            bras.release()
        pca.deinit()