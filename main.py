import os
from dotenv import load_dotenv

from src.control.controller_pid import ControllerPID
from src.control.controller_lqr import ControllerLQR
from src.control.arm import Arm

from src.estimation.estimation import Estimation

from src.perception.perception import Perception
from src.perception.camera import Camera
from src.perception.detector import OrangeDetector
from src.actuator.actuation import Actuation
from src.actuator.Servo import create_pca, Servo

from src.utils import PID, Calibration, StateSpaceModel
import csv
import logging
import math
import time
import numpy as np

# Every run appends a row per tick here. This is what makes PID tuning
# measurable instead of guesswork - plot x_mm/y_mm against t to see
# overshoot, oscillation period and settling time.
now = time.strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = f"tuning_log_{now}.csv"
load_dotenv()

def build_arms(pca) -> np.ndarray:
    arm_1 = Arm(
        servo=Servo(pca, channel=0, name="bras_1", reverse=True, offset_rad=float(os.getenv("OFFSET_ARM_1"))),
        L1=float(os.getenv("L1")), L2=float(os.getenv("L2")), L3=float(os.getenv("L3")), mount_angle_rad=float(os.getenv("MOUNT_ANGLE_ARM_1")),
    )
    arm_2 = Arm(
        servo=Servo(pca, channel=1, name="bras_2", reverse=True, offset_rad=float(os.getenv("OFFSET_ARM_2"))),
        L1=float(os.getenv("L1")), L2=float(os.getenv("L2")), L3=float(os.getenv("L3")), mount_angle_rad=float(os.getenv("MOUNT_ANGLE_ARM_2")),
    )
    arm_3 = Arm(
        servo=Servo(pca, channel=2, name="bras_3", reverse=True, offset_rad=float(os.getenv("OFFSET_ARM_3"))),
        L1=float(os.getenv("L1")), L2=float(os.getenv("L2")), L3=float(os.getenv("L3")), mount_angle_rad=float(os.getenv("MOUNT_ANGLE_ARM_3")),
    )
    return np.array([arm_1, arm_2, arm_3])


PID_X = PID(kp=0.0012,ki=0.0002,kd=0.0009, integral_limits=(-100, 100))
PID_Y = PID(kp=0.0012,ki=0.0002,kd=0.0009, integral_limits=(-100, 100))

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
perception = Perception(calibration=Calibration(origin_px=(int(os.getenv("CAMERA_WIDTH")) // 2, int(os.getenv("CAMERA_HEIGHT")) // 2),
                                                mm_per_px=0.14),
                        camera=Camera(source=int(os.getenv("CAMERA_SOURCE")),width=int(os.getenv("CAMERA_WIDTH")),height=int(os.getenv("CAMERA_HEIGHT")),
                                      framerate=int(os.getenv("CAMERA_FRAMERATE"))),
                        detector=OrangeDetector(min_area=int(os.getenv("DETECTION_MIN_AREA")),score_threshold=float(os.getenv("SCORE_THRESHOLD")))
                        )

estimation = Estimation(process_noise_std=float(os.getenv("PROCESS_NOISE_STD")),measurement_noise_std=float(os.getenv("MEASUREMENT_NOISE_STD")), warmup_ticks=int(os.getenv("WARMUP_TICKS")),max_dropout_seconds=float(os.getenv("MAX_TIMEOUT_SECONDS")))
MAX_TILT_RAD = math.radians(float(os.getenv("MAX_TILT_DEG")))
CONTROL_MODE = os.getenv("CONTROL_MODE")

if CONTROL_MODE == "PID":
    controller = ControllerPID(pid_x=PID_X, pid_y=PID_Y, max_tilt_rad=MAX_TILT_RAD)
elif CONTROL_MODE == "LQR":
    controller = ControllerLQR.from_bryson(
        model=StateSpaceModel(g_eff=float(os.getenv("G_EFF"))),
        max_pos_mm=float(os.getenv("LQR_MAX_POS_MM")),
        max_vel_mm_s=float(os.getenv("LQR_MAX_VEL_MM_S")),
        max_tilt_rad=MAX_TILT_RAD,
    )
    print(f"LQR gain K (rows: roll, pitch; cols: x, y, vx, vy):\n{controller.K}")
else:
    raise ValueError(f"CONTROL_MODE must be 'PID' or 'LQR', got {CONTROL_MODE!r}")


controller.set_target(0.0, 0.0)
actuator = Actuation(arms=arms, plate_radius=float(os.getenv("L")), neutral_height=float(os.getenv("H")), max_rate_rad_s=5.0)

print("INITIALISATION COMPLETE: starting main loop. Press Ctrl+C to stop and write log.")

log_file = open(LOG_PATH, "w", newline="")
log = csv.writer(log_file)
log.writerow(["dt","dt_detection","dt_estimation","dt_control","dt_actuation",
              "valid","x_mm","y_mm","vx","vy","roll","pitch"]) #dt is the global time between each loop

# Centre of the camera frame - the target the ball is driven towards.
FRAME_CX = int(os.getenv("CAMERA_WIDTH")) / 2.0
FRAME_CY = int(os.getenv("CAMERA_HEIGHT")) / 2.0
PRINT_EVERY_S = 0.2  # console is rate-limited; the CSV gets every tick


def log_tick(dt,dt_detection, dt_estimation, dt_control, dt_actuation, state, command):
    log.writerow([dt, dt_detection, dt_estimation, dt_control, dt_actuation,
                  int(state.valid), state.x, state.y, state.vx, state.vy,
                  command.roll, command.pitch])


t0 = time.monotonic()
last_print= t0
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
            dt_actuation,
            estimated_state,
            plate_command,
        )

finally:
    log_file.close()
    print(f"\nWrote {LOG_PATH}")
    actuator.release()
    perception.release()
