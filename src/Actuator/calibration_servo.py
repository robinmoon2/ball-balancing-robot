#!/usr/bin/env python3

import time

from Servo import Servo, create_pca


pca = create_pca()

servos = [
    Servo(pca, channel=0, name="bras_1"),
    Servo(pca, channel=1, name="bras_2"),
    Servo(pca, channel=2, name="bras_3"),
]

try:
    print("Positionnement des servomoteurs à l'angle minimal : 0° / 0 rad")

    for servo in servos:
        servo.set_angle(0)

    time.sleep(2)

except KeyboardInterrupt:
    print("\nArrêt demandé.")

finally:
    for servo in servos:
        servo.release()

    pca.deinit()
    print("Servomoteurs libérés.")