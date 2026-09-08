from control.controller import Controller
from control.arm import Arm

from estimation.estimation import Estimation

from perception.perception import Perception
from perception.camera import Camera
from perception.detector import OrangeDetector
from Actuator.actuation import Actuation
from Actuator.Servo import create_pca, Servo

from utils import PID, Calibration
import logging 
import time
import numpy as np 

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
PID_X = PID(kp=0.3,ki=0.0,kd=0.0)
PID_Y = PID(kp=0.3,ki=0.0,kd=0.0)
max_tilt_rad = np.radians(20)

# Variables for arms

L = 110.0  # plate half-width: center -> spherical joint (mm)
L1 = 95.0  # distal link: elbow -> spherical joint (mm)
L2 = 70.0  # proximal link: motor axis -> elbow (mm)
L3 = 90.0  # base radius: center -> motor axis (mm)

h = 60.0  # global: starting/center plate height (mm) - plate begins here

OFFSET_ARM_1 = 0.66  # rad, from calibration_servo.py
OFFSET_ARM_2 = 0.55
OFFSET_ARM_3 = 0.8

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
perception = Perception(calibration=Calibration(origin_px= [0,0], mm_per_px= 1), 
                        camera=Camera(source=camera_source,width=camera_width,height=camera_height), 
                        detector=OrangeDetector(min_area=detection_min_area,score_threshold=score_threshold)
                        )

estimation = Estimation(process_noise_std=process_noise_std,measurement_noise_std=measurement_noise_std, warmup_ticks=warmup_ticks,max_dropout_seconds=max_timeout_seconds)
controller = Controller(pid_x=PID_X, pid_y=PID_Y,max_tilt_rad=max_tilt_rad)
actuator = Actuation(arms=arms, plate_radius=L, neutral_height=h)

print("initialisation complete")