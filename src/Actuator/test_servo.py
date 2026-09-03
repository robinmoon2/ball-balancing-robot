#!/usr/bin/env python3

import numpy as np
from Servo import Servo, create_pca


def main():
    pca = create_pca(address=0x40, frequency=50)

    servos = {
        1: Servo(pca, channel=0, name="bras_1", reverse=True),
        2: Servo(pca, channel=1, name="bras_2", reverse=True),
        3: Servo(pca, channel=2, name="bras_3", reverse=True),
    }

    try:
        print("Initialisation des servos à 0 rad...")

        for servo in servos.values():
            servo.set_angle(0.0)

        print("\nCommandes :")
        print("- Entrez : numéro_du_servo angle_en_radians")
        print("- Exemple : 2 1.57")
        print("- Utilisez une valeur entre 0 et pi")
        print("- Tapez q pour quitter\n")

        while True:
            commande = input("Commande : ").strip().lower()

            if commande in ("q", "quit", "exit"):
                break

            valeurs = commande.split()

            if len(valeurs) != 2:
                print("Format attendu : servo angle_rad")
                continue

            try:
                numero_servo = int(valeurs[0])
                angle_rad = float(valeurs[1])
            except ValueError:
                print("Le numéro du servo et l'angle doivent être numériques.")
                continue

            if numero_servo not in servos:
                print("Servo invalide. Choisissez 1, 2 ou 3.")
                continue

            if not 0.0 <= angle_rad <= np.pi:
                print(f"L'angle doit être compris entre 0 et {np.pi:.4f} rad.")
                continue

            servo = servos[numero_servo]
            servo.set_angle(angle_rad)

            print(
                f"{servo.name} réglé à "
                f"{angle_rad:.4f} rad "
                f"({np.degrees(angle_rad):.2f}°)"
            )

    except KeyboardInterrupt:
        print("\nArrêt avec Ctrl+C.")

    finally:
        print("\nRetour des servos à 0 rad...")

        for servo in servos.values():
            servo.set_angle(0.0)

        for servo in servos.values():
            servo.release()

        pca.deinit()
        print("Programme terminé.")


if __name__ == "__main__":
    main()