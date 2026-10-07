#!/usr/bin/env python3

import time

from Servo import Servo, create_pca


pca = create_pca()

servos = [
    Servo(pca, channel=0, name="bras_1",reverse=True),
    Servo(pca, channel=1, name="bras_2",reverse=True),
    Servo(pca, channel=2, name="bras_3",reverse=True),
]

try:
    print("Positionnement des servomoteurs à l'angle minimal : 0° / 0 rad")

    for servo in servos:
         servo.set_angle(-0.2)

    time.sleep(60)
except KeyboardInterrupt:
    print("\nArrêt demandé.")

finally:
    for servo in servos:
        servo.release()

    pca.deinit()
    print("Servomoteurs libérés.")
