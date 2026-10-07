from src.control.controller import Controller
from src.control.arm import Arm

from src.estimation.estimation import Estimation

from src.perception.perception import Perception
from src.perception.camera import Camera
from src.perception.detector import OrangeDetector
from src.Actuator.actuation import Actuation
from src.Actuator.Servo import create_pca, Servo

from src.utils import PID, Calibration
import csv
import logging
import time
import numpy as np

# Every run appends a row per tick here. This is what makes PID tuning
# measurable instead of guesswork - plot x_mm/y_mm against t to see
# overshoot, oscillation period and settling time.
now = time.strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = f"tuning_log_{now}.csv"

def build_arms(pca) -> np.ndarray:
    arm_1 = Arm(
        servo=Servo(pca, channel=0, name="bras_1", reverse=True, offset_rad=OFFSET_ARM_1),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_1,
    )
    arm_2 = Arm(
        servo=Servo(pca, channel=1, name="bras_2", reverse=True, offset_rad=OFFSET_ARM_2),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_2,
    )
    arm_3 = Arm(
        servo=Servo(pca, channel=2, name="bras_3", reverse=True, offset_rad=OFFSET_ARM_3),
        L1=L1, L2=L2, L3=L3, mount_angle_rad=MOUNT_ANGLE_ARM_3,
    )
    return np.array([arm_1, arm_2, arm_3])

# Variables for perception
camera_source: int = 0
camera_height: int = 480  # must be a multiple of 16 (I420 plane alignment)
camera_width: int = 640  # must be a multiple of 32
camera_framerate: int = 30
detection_min_area: int = 200
score_threshold: int = 0.15

# Variables for estimation
process_noise_std: float = 1500.0
measurement_noise_std: float = 1.5
warmup_ticks: int = 5
max_timeout_seconds = 0.25

#Variables for control 
PID_X = PID(kp=0.003,ki=0.001,kd=-0.005)
PID_Y = PID(kp=0.003,ki=0.001,kd=-0.005)
max_tilt_rad = np.radians(8)

# Variables for arms

L = 150.0  # plate half-width: center -> spherical joint (mm)
L1 = 116.0  # distal link: elbow -> spherical joint (mm)
L2 = 100.0  # proximal link: motor axis -> elbow (mm)
L3 = 90.0  # base radius: center -> motor axis (mm)

h = 70.0  # global: starting/center plate height (mm) - plate begins here

OFFSET_ARM_1 = 0.4 # rad, from calibration_servo.py
OFFSET_ARM_2 = 0.1
OFFSET_ARM_3 = 0.3

# Physical mount azimuth of each named arm (bras_1/2/3, matching
# calibration_servo.py) - NOT in numeric order: arm 2 sits at 0 deg,
# arm 3 at 120 deg, arm 1 at 240 deg.
MOUNT_ANGLE_ARM_1 = np.radians(240.0)
MOUNT_ANGLE_ARM_2 = np.radians(0.0)
MOUNT_ANGLE_ARM_3 = np.radians(120.0)

pca = create_pca()
arms: np.ndarray = build_arms(pca)

## Initialize components

# Initialize principals components 
# The goal is to hold the ball in the MIDDLE OF THE CAMERA FRAME, so the
# origin is the frame centre by definition - there is no need to locate the
# plate centre in pixels at all. Controller's target of (0,0) then means
# exactly "ball at the centre of the image".
# mm_per_px only sets the loop gain now; measure it properly with
# `python -m src.perception.calibrate` when you want the gains to be in
# real units.
perception = Perception(calibration=Calibration(origin_px=(camera_width / 2, camera_height / 2),
                                                mm_per_px=0.14),
                        camera=Camera(source=camera_source,width=camera_width,height=camera_height,
                                      framerate=camera_framerate),
                        detector=OrangeDetector(min_area=detection_min_area,score_threshold=score_threshold)
                        )

estimation = Estimation(process_noise_std=process_noise_std,measurement_noise_std=measurement_noise_std, warmup_ticks=warmup_ticks,max_dropout_seconds=max_timeout_seconds)
controller = Controller(pid_x=PID_X, pid_y=PID_Y,max_tilt_rad=max_tilt_rad)
controller.set_target(0.0, 0.0)
actuator = Actuation(arms=arms, plate_radius=L, neutral_height=h, max_raate_rad_s=5.0)

print("INITIALISATION COMPLETE: starting main loop. Press Ctrl+C to stop and write log.")

log_file = open(LOG_PATH, "w", newline="")
log = csv.writer(log_file)
log.writerow(["dt","dt_detection","dt_estimation","dt_control","dt_actuation"]) #dt is the global time between each loop

# Centre of the camera frame - the target the ball is driven towards.
FRAME_CX = camera_width / 2.0
FRAME_CY = camera_height / 2.0
PRINT_EVERY_S = 0.2  # console is rate-limited; the CSV gets every tick


def log_tick(dt,dt_detection, dt_estimation, dt_control, dt_actuation):
    log.writerow([dt, dt_detection, dt_estimation, dt_control, dt_actuation])


t0 = time.monotonic()
last_t = time.monotonic()
try:
    while True:
        t = time.monotonic()
        dt = t - last_t
        last_t = t

        # Perception: pixel -> frame-centred mm, or None if no ball this frame.
        t_detection_start = time.monotonic()
        ball_position_mm = perception.read()
        
        t_detection_end = time.monotonic()
        dt_detection = t_detection_end - t_detection_start
        
        t_estimation_start = time.monotonic()
        estimated_state = estimation.update(ball_position_mm, perception.last_capture_t)

        t_estimation_end = time.monotonic()
        dt_estimation = t_estimation_end - t_estimation_start


        # Control: PID -> plate orientation command (roll/pitch)
        t_control_start = time.monotonic()
        plate_command = controller.update(estimated_state, dt)
        t_control_end = time.monotonic()
        dt_control = t_control_end - t_control_start
        # Actuation: only move while the estimate is trustworthy. With no
        # ball (or during warm-up) we hold the last commanded pose rather
        # than re-levelling, so the plate doesn't twitch every time
        # detection blinks. Controller has already reset its integrators,
        # so there's no windup to catch up on when the ball comes back.
        applied = estimated_state.valid
        if applied:
            dt_actuation_start = time.monotonic()
            actuator.apply(plate_command, dt)
            dt_actuation_end = time.monotonic()
            dt_actuation = dt_actuation_end - dt_actuation_start
        else:
            dt_actuation = 0.0
        do_print = (t - last_print) >= PRINT_EVERY_S
        if do_print:
            last_print = t
        log_tick(
            dt,
            dt_detection,
            dt_estimation,
            dt_control,
            dt_actuation
        )

finally:
    log_file.close()
    print(f"\nWrote {LOG_PATH}")
    actuator.release()
    perception.release()