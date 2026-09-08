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
LOG_PATH = "tuning_log.csv"

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
camera_height: int = 480
camera_width: int = 640
detection_min_area: int = 200
score_threshold: int = 0.15

# Variables for estimation
process_noise_std: float = 1500.0
measurement_noise_std: float = 1.5
warmup_ticks: int = 5
max_timeout_seconds = 0.25

#Variables for control 
PID_X = PID(kp=0.0035,ki=0.0,kd=0.0008)
PID_Y = PID(kp=0.0035,ki=0.0,kd=0.0008)
# Per-axis limit. Roll and pitch clamp independently, so the worst-case
# COMBINED tilt is sqrt(2)x this - at 20 deg that was 28 deg, which the arms
# cannot reach and Actuation rightly refused. 8 deg keeps the combined worst
# case at ~11 deg, reachable at both h=60 and h=130. Raise it once h is
# measured and you know the real workspace.
max_tilt_rad = np.radians(8)

# Variables for arms

L = 110.0  # plate half-width: center -> spherical joint (mm)
L1 = 95.0  # distal link: elbow -> spherical joint (mm)
L2 = 70.0  # proximal link: motor axis -> elbow (mm)
L3 = 90.0  # base radius: center -> motor axis (mm)

h = 60.0  # global: starting/center plate height (mm) - plate begins here

OFFSET_ARM_1 = 0.66+0.66  # rad, from calibration_servo.py
OFFSET_ARM_2 = 0.55+0.55
OFFSET_ARM_3 = 0.8+0.75

# Physical mount azimuth of each named arm (bras_1/2/3, matching
# calibration_servo.py) - NOT in numeric order: arm 2 sits at 0 deg,
# arm 3 at 120 deg, arm 1 at 240 deg.
MOUNT_ANGLE_ARM_1 = np.radians(240.0)
MOUNT_ANGLE_ARM_2 = np.radians(0.0)
MOUNT_ANGLE_ARM_3 = np.radians(120.0)

pca = create_pca()
arms: np.ndarray = build_arms(pca)

print("Starting main.py")

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
                        camera=Camera(source=camera_source,width=camera_width,height=camera_height),
                        detector=OrangeDetector(min_area=detection_min_area,score_threshold=score_threshold)
                        )

estimation = Estimation(process_noise_std=process_noise_std,measurement_noise_std=measurement_noise_std, warmup_ticks=warmup_ticks,max_dropout_seconds=max_timeout_seconds)
controller = Controller(pid_x=PID_X, pid_y=PID_Y,max_tilt_rad=max_tilt_rad)
controller.set_target(0.0, 0.0)
actuator = Actuation(arms=arms, plate_radius=L, neutral_height=h)

print("initialisation complete")

log_file = open(LOG_PATH, "w", newline="")
log = csv.writer(log_file)
log.writerow(["t", "dt", "found", "px", "py", "frac_x", "frac_y",
              "x_mm", "y_mm", "vx", "vy", "valid",
              "roll_cmd", "pitch_cmd", "applied",
              "arm1_delta", "arm2_delta", "arm3_delta"])

# Centre of the camera frame - the target the ball is driven towards.
FRAME_CX = camera_width / 2.0
FRAME_CY = camera_height / 2.0
PRINT_EVERY_S = 0.2  # console is rate-limited; the CSV gets every tick


def log_tick(t_rel, dt, detection, state, command, deltas, applied, do_print):
    """Record one tick: where the ball is, and what the arms did about it.

    The ball is reported three ways - raw pixels, as a fraction of the
    camera's own limits (0 = centre of frame, +/-1 = edge of frame), and in
    mm via the calibration. The arm action is each arm's movement away from
    its own calibrated flat position, so 0,0,0 means a level plate.
    """
    if detection.found:
        frac_x = (detection.x - FRAME_CX) / FRAME_CX
        frac_y = (detection.y - FRAME_CY) / FRAME_CY
        where = (f"ball px=({detection.x:4d},{detection.y:4d}) "
                 f"frame=({frac_x:+.2f},{frac_y:+.2f}) "
                 f"mm=({state.x:+7.1f},{state.y:+7.1f})")
    else:
        frac_x = frac_y = float("nan")
        where = f"ball NOT FOUND{'':>44}"

    if applied:
        action = "arms " + " ".join(f"{d:+.3f}" for d in deltas)
    else:
        action = "arms HOLD (estimate not valid)"

    if do_print:
        print(f"{where} | roll={command.roll:+.3f} pitch={command.pitch:+.3f} | {action}")

    log.writerow([
        f"{t_rel:.4f}", f"{dt:.4f}", int(detection.found),
        detection.x, detection.y, f"{frac_x:.4f}", f"{frac_y:.4f}",
        f"{state.x:.2f}", f"{state.y:.2f}",
        f"{state.vx:.1f}", f"{state.vy:.1f}", int(state.valid),
        f"{command.roll:.4f}", f"{command.pitch:.4f}", int(applied),
        *(f"{d:.4f}" for d in deltas),
    ])


t0 = time.monotonic()
last_print = 0.0
last_t = time.monotonic()
try:
    while True:
        t = time.monotonic()
        dt = t - last_t
        last_t = t

        # Perception: pixel -> frame-centred mm, or None if no ball this frame.
        ball_position_mm = perception.read()

        # Estimation: ball position in mm, with velocity and a validity flag.
        estimated_state = estimation.update(ball_position_mm, t)

        # Control: PID -> plate orientation command (roll/pitch)
        plate_command = controller.update(estimated_state, dt)

        # Actuation: only move while the estimate is trustworthy. With no
        # ball (or during warm-up) we hold the last commanded pose rather
        # than re-levelling, so the plate doesn't twitch every time
        # detection blinks. Controller has already reset its integrators,
        # so there's no windup to catch up on when the ball comes back.
        applied = estimated_state.valid
        if applied:
            actuator.apply(plate_command, dt)

        do_print = (t - last_print) >= PRINT_EVERY_S
        if do_print:
            last_print = t
        log_tick(
            t_rel=t - t0,
            dt=dt,
            detection=perception.last_detection,
            state=estimated_state,
            command=plate_command,
            deltas=actuator.last_deltas,
            applied=applied,
            do_print=do_print,
        )

        time.sleep(0.01)  # Loop delay to prevent CPU overload
finally:
    log_file.close()
    print(f"\nWrote {LOG_PATH}")
    actuator.release()
    perception.release()