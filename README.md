# ball-balancing-robot

This project is made for a ball balancing robot RRR using centroid image based.

The goals was to experiment control engineering and to compare PIDs control and LQR control with an estimation using a simple kalman filter.

## File description

This project is written on python using the uv environnment the overall running on a raspberry pi 4.

- **/src** : contains all the code file for the robot. In it we can find the folder for the actuators, the estimatiorn with the kalman filter, the control including the inverse kinematics model and the control mode ( PID or LQR ), also the perception with the camera algorithm with centroid identification.

- **tuning_analyse_log.ipynb**: A jupyter notebook used to make analysiss after a run of the robot using the log of the main file.

- **/tool** : a folder with hte geometric explorer that allow us to make the geomatry varies and gives us different graph for it to figure out how can we improve the structure and with the values.

[!WARNING]
This is only a geometric function, no link with the choose of the material for the parts

## Hardware

### Electronics

The computer inside is a raspberry 4. Connected to it we have a raspberry pi camera that is turning up to see the ball in movement. Then we have the PCA9685 that is link using I2C to the raspberry pi and help us drive the servo motors with more power and better stability. For the servo motors, there is 3 of MZ996 from Miuzei, 180 degrees metal.

[!NOTE]
Some people will think that the servos are too much for the project. The goal was to build the kinematic part knowing that the hardware will be not a limited factor for the project
