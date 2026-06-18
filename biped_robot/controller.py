import mujoco
import mujoco.viewer
import numpy as np
import time

from .config import MODEL_PATH, PLAYBACK_SPEED, TRAJECTORY_FLUSH_EVERY_STEPS, TRAJECTORY_SAMPLE_EVERY_STEPS
from .gains import get_gains
from .input import ViewerCommand
from .kinematics import body_id, get_imu_data
from .model_constants import FOOT_GEOM_NAMES, JOINT_NAMES, TRAJECTORY_BODY_NAMES
from .trajectory_logger import TrajectoryLogger

# ==========================================================
# LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Starting Bipedal Walking Controller")
print("=" * 50)
print("  VIEWER CONTROLS:")
print("    W / Up Arrow       -  Walk Forward")
print("    S / Down Arrow     -  Walk Backward")
print("    A / Left Arrow     -  Turn Left")
print("    D / Right Arrow    -  Turn Right")
print("    Space / Esc        -  Stop")
print("=" * 50)

# ==========================================================
# JOINT LIST & HELPERS
# ==========================================================

joint_names = JOINT_NAMES

def get_torso_height():
    return float(data.xpos[body_id(model, "torso"), 2])

def get_com_y():
    return float(data.subtree_com[body_id(model, "torso"), 1])

def get_foot_y(side):
    name = "left_foot_link" if side == 'L' else "right_foot_link"
    return float(data.xpos[body_id(model, name), 1])

# ==========================================================
# GAIT PARAMETERS
# ==========================================================

angle = 0.19

# ==========================================================
# FORWARD GAIT KEYFRAMES
# ==========================================================

sit = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -1.0, "left_knee_pitch": 1.8, "left_ankle_pitch": -0.8, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.0, "right_knee_pitch": 1.8, "right_ankle_pitch": -0.8, "right_ankle_roll": 0.0
}

stand = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0
}

lean_L_initial = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

lift_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.8, "right_knee_pitch": 1.2, "right_ankle_pitch": -0.4, "right_ankle_roll": angle
}

plant_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.5, "right_knee_pitch": 0.3, "right_ankle_pitch": 0.2, "right_ankle_roll": angle
}

center_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.025, "left_knee_pitch": 0.15, "left_ankle_pitch": -0.175, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.325, "right_knee_pitch": 0.3, "right_ankle_pitch": 0.025, "right_ankle_roll": 0.0
}

lean_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.20, "left_knee_pitch": 0.0, "left_ankle_pitch": -0.20, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

lift_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.8, "left_knee_pitch": 1.2, "left_ankle_pitch": -0.4, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

plant_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.5, "left_knee_pitch": 0.3, "left_ankle_pitch": 0.2, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

center_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.325, "left_knee_pitch": 0.3, "left_ankle_pitch": 0.025, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.025, "right_knee_pitch": 0.15, "right_ankle_pitch": -0.175, "right_ankle_roll": 0.0
}

lean_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": 0.20, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.20, "right_ankle_roll": angle
}

# ==========================================================
# BACKWARD GAIT KEYFRAMES (conservative 30% reversed stride)
# ==========================================================

# Right leg lifts and swings generously backward with massive clearance
lift_R_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.10, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.20, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.30, "right_knee_pitch": 1.20, "right_ankle_pitch": -0.50, "right_ankle_roll": angle
}

# Right foot plants exactly 11.8cm behind
plant_R_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.10, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.20, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": 0.05, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.35, "right_ankle_roll": angle
}

# Body shifts backward (front leg starts straightening to push smoothly)
center_R_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.10, "left_knee_pitch": 0.15, "left_ankle_pitch": -0.05, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.025, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.275, "right_ankle_roll": 0.0
}

# Weight fully on right leg (front leg fully straight to balance on one leg)
lean_R_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.10, "left_knee_pitch": 0.00, "left_ankle_pitch": 0.10, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.10, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.20, "right_ankle_roll": -angle
}

# Left leg lifts and swings generously backward with massive clearance
lift_L_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.30, "left_knee_pitch": 1.20, "left_ankle_pitch": -0.50, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.10, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.20, "right_ankle_roll": -angle
}

# Left foot plants exactly 11.8cm behind
plant_L_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.05, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.35, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.10, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.20, "right_ankle_roll": -angle
}

# Body shifts backward (front leg starts straightening to push smoothly)
center_L_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.025, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.275, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.10, "right_knee_pitch": 0.15, "right_ankle_pitch": -0.05, "right_ankle_roll": 0.0
}

# Weight fully on left leg (front leg fully straight to balance on one leg)
lean_L_bk = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.10, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.20, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.10, "right_knee_pitch": 0.00, "right_ankle_pitch": 0.10, "right_ankle_roll": angle
}

# ==========================================================
# TURN LEFT GAIT KEYFRAMES
# ==========================================================

lean_L_initial_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

lift_R_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.80, "right_knee_pitch": 1.20, "right_ankle_pitch": -0.40, "right_ankle_roll": angle
}

plant_R_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

center_R_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.0, "left_knee_pitch": 0.0, "left_ankle_pitch": 0.0, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0
}

lean_R_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.0, "left_knee_pitch": 0.0, "left_ankle_pitch": 0.0, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

lift_L_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.80, "left_knee_pitch": 1.20, "left_ankle_pitch": -0.40, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

plant_L_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

center_L_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.0, "right_knee_pitch": 0.0, "right_ankle_pitch": 0.0, "right_ankle_roll": 0.0
}

lean_L_turn_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": 0.0, "right_knee_pitch": 0.0, "right_ankle_pitch": 0.0, "right_ankle_roll": angle
}

# ==========================================================
# TURN RIGHT GAIT KEYFRAMES
# ==========================================================

lean_L_initial_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

lift_R_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.80, "right_knee_pitch": 1.20, "right_ankle_pitch": -0.40, "right_ankle_roll": angle
}

plant_R_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

center_R_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.0, "left_knee_pitch": 0.0, "left_ankle_pitch": 0.0, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0
}

lean_R_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.0, "left_knee_pitch": 0.0, "left_ankle_pitch": 0.0, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

lift_L_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.80, "left_knee_pitch": 1.20, "left_ankle_pitch": -0.40, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

plant_L_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

center_L_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.0, "right_knee_pitch": 0.0, "right_ankle_pitch": 0.0, "right_ankle_roll": 0.0
}

lean_L_turn_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.30, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": 0.0, "right_knee_pitch": 0.0, "right_ankle_pitch": 0.0, "right_ankle_roll": angle
}

# ==========================================================
# STATE MACHINE
# ==========================================================
# 0:  SIT
# 1:  STAND (sit -> stand)
# 10: IDLE  (standing still, waiting for viewer key command)
# 2:  SHIFT_INITIAL (stand -> lean_L_initial)
# 3:  LIFT_RIGHT
# 4:  PLANT_RIGHT
# 50: SHIFT_R_PHASE1
# 51: SHIFT_R_PHASE2
# 6:  LIFT_LEFT
# 7:  PLANT_LEFT
# 80: SHIFT_L_PHASE1
# 81: SHIFT_L_PHASE2
# 90: SETTLE_FROM_R (lean_R -> stand, stopping walk)
# 91: SETTLE_FROM_L (lean_L -> stand, stopping walk)
# 99: FALLEN

state = 0
state_start = 0.0
step_count = 0

# Movement state
walk_mode = "idle"   # "idle", "forward", "backward"

MIN_DURATIONS = {
    0: 0.5,
    1: 1.0,     # stand up
    10: 999,    # idle (stays until key pressed)
    2: 1.2,     # initial shift
    3: 0.3,     # lift right
    4: 0.2,     # plant right
    50: 0.8,    # shift R 1
    51: 0.8,    # shift R 2
    6: 0.3,     # lift left
    7: 0.2,     # plant left
    80: 0.8,    # shift L 1
    81: 0.8,    # shift L 2
    90: 1.0,    # settling from right lean
    91: 1.0,    # settling from left lean
    99: 999,    # fallen
}

# ==========================================================
# KEYBOARD INPUT
# ==========================================================

viewer_command = ViewerCommand()

# ==========================================================
# HELPERS
# ==========================================================

def check_com_over_foot(side, threshold=0.05):
    com_pos = data.qpos[:3]
    if side == 'L':
        foot_pos = data.geom("left_foot").xpos
    else:
        foot_pos = data.geom("right_foot").xpos
        
    dx = com_pos[0] - foot_pos[0]
    dy = com_pos[1] - foot_pos[1]
    
    # Project into local coordinates using yaw
    yaw = unwrapped_yaw
    local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

    if side == 'L':
        return local_y > -threshold
    else:
        return local_y < threshold

unwrapped_yaw = 0.0
last_raw_yaw = None

def update_yaw():
    global unwrapped_yaw, last_raw_yaw
    _, _, raw_yaw = get_imu_data(model, data)
    if last_raw_yaw is None:
        last_raw_yaw = raw_yaw
        unwrapped_yaw = raw_yaw
        return
    diff = raw_yaw - last_raw_yaw
    if diff > np.pi: diff -= 2*np.pi
    elif diff < -np.pi: diff += 2*np.pi
    unwrapped_yaw += diff
    last_raw_yaw = raw_yaw

def is_fallen():
    roll, pitch, yaw = get_imu_data(model, data)
    return abs(roll) > 0.8 or abs(pitch) > 0.8

# ==========================================================
# POSE LOOKUP
# ==========================================================

def get_forward_poses(st):
    if st == 2: return stand, lean_L_initial
    if st == 3: return (lean_L_initial if step_count == 0 else lean_L), lift_R
    if st == 4: return lift_R, plant_R
    if st == 50: return plant_R, center_R
    if st == 51: return center_R, lean_R
    if st == 6: return lean_R, lift_L
    if st == 7: return lift_L, plant_L
    if st == 80: return plant_L, center_L
    if st == 81: return center_L, lean_L
    if st == 90: return lean_R, stand
    if st == 91: return lean_L, stand
    return stand, stand

def get_backward_poses(st):
    if st == 2: return stand, lean_L_initial
    if st == 3: return (lean_L_initial if step_count == 0 else lean_L_bk), lift_R_bk
    if st == 4: return lift_R_bk, plant_R_bk
    if st == 50: return plant_R_bk, center_R_bk
    if st == 51: return center_R_bk, lean_R_bk
    if st == 6: return lean_R_bk, lift_L_bk
    if st == 7: return lift_L_bk, plant_L_bk
    if st == 80: return plant_L_bk, center_L_bk
    if st == 81: return center_L_bk, lean_L_bk
    if st == 90: return lean_R_bk, stand
    if st == 91: return lean_L_bk, stand
    return stand, stand

def get_turn_left_poses(st):
    if st == 2: return stand, lean_L_initial_turn_L
    if st == 3: return (lean_L_initial_turn_L if step_count == 0 else lean_L_turn_L), lift_R_turn_L
    if st == 4: return lift_R_turn_L, plant_R_turn_L
    if st == 50: return plant_R_turn_L, center_R_turn_L
    if st == 51: return center_R_turn_L, lean_R_turn_L
    if st == 6: return lean_R_turn_L, lift_L_turn_L
    if st == 7: return lift_L_turn_L, plant_L_turn_L
    if st == 80: return plant_L_turn_L, center_L_turn_L
    if st == 81: return center_L_turn_L, lean_L_turn_L
    if st == 90: return lean_R_turn_L, stand
    if st == 91: return lean_L_turn_L, stand
    return stand, stand

def get_turn_right_poses(st):
    if st == 2: return stand, lean_L_initial_turn_R
    if st == 3: return (lean_L_initial_turn_R if step_count == 0 else lean_L_turn_R), lift_R_turn_R
    if st == 4: return lift_R_turn_R, plant_R_turn_R
    if st == 50: return plant_R_turn_R, center_R_turn_R
    if st == 51: return center_R_turn_R, lean_R_turn_R
    if st == 6: return lean_R_turn_R, lift_L_turn_R
    if st == 7: return lift_L_turn_R, plant_L_turn_R
    if st == 80: return plant_L_turn_R, center_L_turn_R
    if st == 81: return center_L_turn_R, lean_L_turn_R
    if st == 90: return lean_R_turn_R, stand
    if st == 91: return lean_L_turn_R, stand
    return stand, stand

def get_poses_for_state(st):
    if st == 0: return sit, sit
    if st == 1: return sit, stand
    if st == 10: return stand, stand
    if walk_mode == "backward":
        return get_backward_poses(st)
    elif walk_mode == "turn_left":
        return get_turn_left_poses(st)
    elif walk_mode == "turn_right":
        return get_turn_right_poses(st)
    else:
        return get_forward_poses(st)

# ==========================================================
# INITIAL POSE
# ==========================================================

def apply_sitting_pose():
    data.qpos[2] = 0.90
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = sit[jn]
    mujoco.mj_forward(model, data)

apply_sitting_pose()

# ==========================================================
# SIMULATION LOOP
# ==========================================================

start_time_sim = data.time
last_sim_time = data.time
last_print = 0.0
last_mode_print = ""

step_counter = 0
trajectory_logger = TrajectoryLogger(model, data, joint_names, TRAJECTORY_BODY_NAMES, FOOT_GEOM_NAMES)

with mujoco.viewer.launch_passive(model, data, key_callback=viewer_command.handle_key) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    target_yaw = 0.0

    current_p1 = sit
    current_p2 = sit
    old_state = -1

    left_bias = 0.0
    right_bias = 0.0
    roll_bias = 0.0

    while viewer.is_running():
        step_start = time.time()
        update_yaw()

        # Reset detection
        if data.time < last_sim_time:
            print("Reset detected")
            apply_sitting_pose()
            start_time_sim = data.time
            state = 0
            state_start = 0.0
            step_count = 0
            walk_mode = "idle"
            current_p1 = sit
            current_p2 = sit
            old_state = -1
            left_bias = 0.0
            right_bias = 0.0
            roll_bias = 0.0
        last_sim_time = data.time

        elapsed = data.time - start_time_sim
        state_elapsed = elapsed - state_start
        min_dur = MIN_DURATIONS.get(state, 1.0)
        
        # Accelerate state 2 to eliminate 2s input delay when turning
        if walk_mode in ["turn_left", "turn_right"] and state == 2:
            min_dur = 0.30

        # --- UPDATE VIEWER COMMAND ---
        key_up = viewer_command.key_up
        key_down = viewer_command.key_down
        key_left = viewer_command.key_left
        key_right = viewer_command.key_right
        motion_command = viewer_command.motion_command

        # --- ANTI-SLIP STEERING LOGIC ---
        # To prevent feet from sliding on the ground, we ONLY steer the swinging leg.
        # The planted leg's bias is set to 0.0, which forces the Torso to naturally pivot over the planted foot!
        target_steering = 0.0
        if walk_mode in ["forward", "backward"]:
            if key_left: target_steering = 0.35  # 20 degrees
            elif key_right: target_steering = -0.35
        elif walk_mode in ["turn_left", "turn_right"]:
            if key_left: target_steering = 0.70  # 40 degrees for faster in-place pivoting
            elif key_right: target_steering = -0.70

        target_left_bias = 0.0
        target_right_bias = 0.0

        if walk_mode in ["forward", "backward", "turn_left", "turn_right"]:
            # Right step (States 3, 4, 50, 51)
            # Exclude state 2 (initial lean L) so we don't twist the right leg while it's still planted!
            if state in [3, 4, 50, 51]:
                # If steering Right (-), Right leg swings OUTWARD.
                # If steering Left (+), Right leg swings INWARD (Collision risk! Force 0.0)
                if target_steering < 0: target_right_bias = target_steering
                else: target_right_bias = 0.0
                
            # Left step (States 6, 7, 80, 81)
            elif state in [6, 7, 80, 81]:
                # If steering Left (+), Left leg swings OUTWARD.
                # If steering Right (-), Left leg swings INWARD (Collision risk! Force 0.0)
                if target_steering > 0: target_left_bias = target_steering
                else: target_left_bias = 0.0

        # Smooth interpolation
        left_bias = 0.90 * left_bias + 0.10 * target_left_bias
        right_bias = 0.90 * right_bias + 0.10 * target_right_bias

        # --- FALL DETECTION ---
        if state >= 2 and state != 99 and state != 90 and state != 91 and is_fallen():
            print(f"  *** FALLEN at t={elapsed:.1f}s in state {state} mode={walk_mode} ***")
            state = 99

        # --- IDLE STATE (special handling, bypasses normal transition) ---
        if state == 10:
            if key_up:
                walk_mode = "forward"
                state = 2
            elif key_down:
                walk_mode = "backward"
                state = 2
            elif key_left:
                walk_mode = "turn_left"
                state = 2
                target_yaw = unwrapped_yaw + (np.pi / 6)
            elif key_right:
                walk_mode = "turn_right"
                state = 2
                target_yaw = unwrapped_yaw - (np.pi / 6)

        # --- NORMAL STATE TRANSITIONS ---
        can_transition = state_elapsed >= min_dur

        # COM checks for shift-to-foot states
        com_threshold = 0.08 if walk_mode == "backward" else 0.05
        
        # Only relax for diagonal steering in forward/backward walking.
        # Stationary turns do not step diagonally, so they must use the strict threshold.
        if walk_mode in ["forward", "backward"]:
            if abs(left_bias) > 0.05 or abs(right_bias) > 0.05:
                com_threshold = 0.20
        elif walk_mode in ["turn_left", "turn_right"]:
            com_threshold = 0.20
        
        if state == 2 and can_transition:
            if not check_com_over_foot('L', com_threshold):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True
        elif state == 51 and can_transition:
            if not check_com_over_foot('R', com_threshold):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True
        elif state == 81 and can_transition:
            if not check_com_over_foot('L', com_threshold):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True

        if can_transition and state != 99 and state != 10:
            if state == 0: state = 1
            elif state == 1:
                state = 10
                walk_mode = "idle"
            elif state == 2: state = 3
            elif state == 3: state = 4
            elif state == 4:
                state = 50
                step_count += 1
            elif state == 50: state = 51
            elif state == 51:
                # Decision point: continue walking or stop?
                if walk_mode == "forward" and key_up: state = 6
                elif walk_mode == "backward" and key_down: state = 6
                elif walk_mode == "turn_left" and key_left: state = 6
                elif walk_mode == "turn_right" and key_right: state = 6
                else: state = 90  # Settle and stop
            elif state == 6: state = 7
            elif state == 7:
                state = 80
                step_count += 1
            elif state == 80: state = 81
            elif state == 81:
                # Decision point: continue walking or stop?
                if walk_mode == "forward" and key_up: state = 3
                elif walk_mode == "backward" and key_down: state = 3
                elif walk_mode == "turn_left" and key_left: state = 3
                elif walk_mode == "turn_right" and key_right: state = 3
                else: state = 91  # Settle and stop
            elif state == 90:
                state = 10
                walk_mode = "idle"
                step_count = 0
            elif state == 91:
                state = 10
                walk_mode = "idle"
                step_count = 0

        if state != old_state:
            state_start = elapsed
            if old_state != -1 and state not in [10, 90, 91]:
                if state == 2:
                    print(f"  t={elapsed:.1f}s  IDLE->WALK_{walk_mode.upper()}")
                else:
                    print(f"  t={elapsed:.1f}s  {old_state}->{state}  mode={walk_mode}  steps={step_count}")
            elif state in [90, 91]:
                print(f"  t={elapsed:.1f}s  SETTLING")
            elif state == 10 and old_state != -1 and old_state != 1:
                print(f"  t={elapsed:.1f}s  IDLE (stopped walking)")
                
            current_p1 = current_p2
            if state != 99:
                _, current_p2 = get_poses_for_state(state)
            old_state = state

        # --- COMPUTE TARGET ---
        if state == 99:
            p1, p2 = stand, stand
            alpha = 1.0
        else:
            p1 = current_p1
            p2 = current_p2
            alpha = float(np.clip(state_elapsed / min_dur, 0.0, 1.0))
            alpha = 3 * alpha**2 - 2 * alpha**3

        # --- PD CONTROL ---
        target_joint_positions = {}
        for actuator_id, joint_name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]

            current_pos = data.qpos[qpos_addr]
            current_vel = data.qvel[qvel_addr]
            target_pos = (1.0 - alpha) * p1[joint_name] + alpha * p2[joint_name]

            # Dynamic steering for curving trajectories
            if walk_mode in ["forward", "backward", "turn_left", "turn_right"]:
                if joint_name == "left_hip_yaw":
                    target_pos += left_bias
                elif joint_name == "right_hip_yaw":
                    target_pos += right_bias

            target_joint_positions[joint_name] = target_pos
            kp, kd = get_gains(joint_name, walk_mode, state)
            torque = kp * (target_pos - current_pos) - kd * current_vel
            gear = model.actuator_gear[actuator_id, 0]
            data.ctrl[actuator_id] = np.clip(torque / gear, -1.0, 1.0)

        mujoco.mj_step(model, data)
        viewer.sync()

        # --- DIAGNOSTICS ---
        if elapsed - last_print > 1.0 and state >= 2 and state != 99 and state != 10:
            roll, pitch, _ = get_imu_data(model, data)
            mode_str = f"mode={walk_mode}"
            print(f"    [diag] t={elapsed:.1f}s st={state} {mode_str} roll={roll:.3f} pitch={pitch:.3f}")
            last_print = elapsed

        # --- TELEMETRY (100Hz) ---
        if step_counter % TRAJECTORY_SAMPLE_EVERY_STEPS == 0:
            # Determine fine-grained action for telemetry grouping
            current_action = "idle"
            if walk_mode == "forward":
                if key_left: current_action = "forward_left"
                elif key_right: current_action = "forward_right"
                else: current_action = "forward"
            elif walk_mode == "backward":
                if key_left: current_action = "backward_left"
                elif key_right: current_action = "backward_right"
                else: current_action = "backward"
            elif walk_mode == "turn_left":
                current_action = "turn_left"
            elif walk_mode == "turn_right":
                current_action = "turn_right"

            trajectory_logger.record(
                elapsed,
                state,
                walk_mode,
                motion_command,
                step_count,
                target_joint_positions,
                current_action,
            )
            if step_counter % TRAJECTORY_FLUSH_EVERY_STEPS == 0:
                trajectory_logger.flush()

        step_counter += 1

        remaining = (model.opt.timestep / PLAYBACK_SPEED) - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

trajectory_logger.close()
