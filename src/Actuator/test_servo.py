#!/usr/bin/env python3

import numpy as np
from Servo import Servo, create_pca

OFFSET_ARM_1 = 0.66  # rad
OFFSET_ARM_2 = 0.55  # rad
OFFSET_ARM_3 = 0.8   # rad


def main():
    pca = create_pca(address=0x40, frequency=50)

    servos = {
        1: Servo(pca, channel=0, name="bras_1", reverse=True, offset_rad=OFFSET_ARM_1),
        2: Servo(pca, channel=1, name="bras_2", reverse=True, offset_rad=OFFSET_ARM_2),
        3: Servo(pca, channel=2, name="bras_3", reverse=True, offset_rad=OFFSET_ARM_3),
    }

    try:
        print("Initialisation des servos à 0 rad (neutre)...")
        for servo in servos.values():
            servo.set_angle(0.0)

        print("\nPlage de commande par bras :")
        for num, servo in servos.items():
            print(
                f"  bras_{num} : [{servo.range_min:.2f}, {servo.range_max:.2f}] rad"
            )

        print("\nCommandes :")
        print("  numéro angle   → ex: 2 -0.3")
        print("  offset num val → ex: offset 1 0.7")
        print("  flat           → tous les servos à 0 (neutre)")
        print("  q              → quitter\n")

        while True:
            commande = input("Commande : ").strip().lower()

            if commande in ("q", "quit", "exit"):
                break

            if commande == "flat":
                for servo in servos.values():
                    servo.set_angle(0.0)
                print("Tous les servos à 0 (neutre).")
                continue

            valeurs = commande.split()

            if len(valeurs) == 3 and valeurs[0] == "offset":
                try:
                    numero = int(valeurs[1])
                    new_offset = float(valeurs[2])
                except ValueError:
                    print("Format : offset numéro_servo valeur_rad")
                    continue
                if numero not in servos:
                    print("Servo invalide. Choisissez 1, 2 ou 3.")
                    continue
                servos[numero].offset_rad = new_offset
                servos[numero].set_angle(servos[numero].get_angle())
                print(
                    f"Offset bras_{numero} = {new_offset:.4f} rad  "
                    f"plage : [{servos[numero].range_min:.2f}, "
                    f"{servos[numero].range_max:.2f}]"
                )
                continue

            if len(valeurs) != 2:
                print("Format attendu : servo angle_rad")
                continue

            try:
                numero_servo = int(valeurs[0])
                angle_rad = float(valeurs[1])
            except ValueError:
                print("Valeurs numériques attendues.")
                continue

            if numero_servo not in servos:
                print("Servo invalide. Choisissez 1, 2 ou 3.")
                continue

            servo = servos[numero_servo]

            if not servo.range_min <= angle_rad <= servo.range_max:
                print(
                    f"Hors plage pour {servo.name} : "
                    f"[{servo.range_min:.2f}, {servo.range_max:.2f}] rad"
                )
                continue

            servo.set_angle(angle_rad)
            print(
                f"{servo.name} → {angle_rad:+.4f} rad "
                f"({np.degrees(angle_rad):+.1f}°)  "
                f"[brut: {servo.offset_rad + angle_rad:.4f} rad]"
            )

    except KeyboardInterrupt:
        print("\nArrêt avec Ctrl+C.")

    finally:
        print("\nRetour au neutre...")
        for servo in servos.values():
            servo.set_angle(0.0)
        for servo in servos.values():
            servo.release()
        pca.deinit()

        print("\nOffsets finaux à reporter :")
        for num, servo in servos.items():
            print(f"  OFFSET_ARM_{num} = {servo.offset_rad}")
        print("Programme terminé.")


if __name__ == "__main__":
    main()