import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import math
import collections

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# ==========================================================
# CONFIGURATION
# ==========================================================
PLAYBACK_SPEED = 1.5  # Multiplier for simulation viewer speed (1.0 = real-time, 2.0 = fast forward)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path("biped.xml")
data = mujoco.MjData(model)

print("Starting Bipedal Walking Controller")
print("=" * 50)
if HAS_KEYBOARD:
    print("  KEYBOARD CONTROLS:")
    print("    Up Arrow    -  Walk Forward")
    print("    Down Arrow  -  Walk Backward")
    print("    No keys     -  Stand Still")
else:
    print("  WARNING: 'keyboard' not found. Install: pip install keyboard")
    print("  Robot will walk forward automatically.")
print("=" * 50)

# ==========================================================
# JOINT LIST & HELPERS
# ==========================================================

joint_names = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
]

def get_body_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def get_torso_height():
    return float(data.xpos[get_body_id("torso"), 2])

def get_com_y():
    return float(data.subtree_com[get_body_id("torso"), 1])

def get_foot_y(side):
    name = "left_foot_link" if side == 'L' else "right_foot_link"
    return float(data.xpos[get_body_id(name), 1])

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
# STATE MACHINE
# ==========================================================
# 0:  SIT
# 1:  STAND (sit -> stand)
# 10: IDLE  (standing still, waiting for keyboard input)
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

key_up = False
key_down = False
key_left = False
key_right = False

def update_keys():
    global key_up, key_down, key_left, key_right
    if not HAS_KEYBOARD:
        return
    try:
        key_up = keyboard.is_pressed('up')
        key_down = False
        key_left = False
        key_right = False
    except:
        pass

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

def quat_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def get_imu_data():
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    adr = model.sensor_adr[sensor_id]
    quat = data.sensordata[adr:adr+4]
    return quat_to_euler(quat)

unwrapped_yaw = 0.0
last_raw_yaw = None

def update_yaw():
    global unwrapped_yaw, last_raw_yaw
    _, _, raw_yaw = get_imu_data()
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
    roll, pitch, yaw = get_imu_data()
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

def get_poses_for_state(st):
    if st == 0: return sit, sit
    if st == 1: return sit, stand
    if st == 10: return stand, stand
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
# PD GAINS
# ==========================================================

KP = {"yaw": 1000, "roll": 1500, "hip": 1200, "knee": 1500, "ankle_pitch": 1000, "ankle_roll": 1000}
KD = {"yaw": 60, "roll": 250, "hip": 150, "knee": 180, "ankle_pitch": 150, "ankle_roll": 150}

def get_gains(name):
    if "yaw" in name: return 50.0, 5.0
    if "hip_roll" in name: return KP["roll"], KD["roll"]
    if "hip_pitch" in name: return KP["hip"], KD["hip"]
    if "knee" in name: return KP["knee"], KD["knee"]
    if "ankle_pitch" in name:
        # Dynamic Compliance: Relax the trailing ankle ONLY during forward walking
        if walk_mode == "forward":
            if "left" in name and state in [50, 51, 6]:
                return 100, 10
            if "right" in name and state in [80, 81, 3]:
                return 100, 10
        return KP["ankle_pitch"], KD["ankle_pitch"]
    if "ankle_roll" in name: return KP["ankle_roll"], KD["ankle_roll"]
    return 100, 10

# ==========================================================
# SIMULATION LOOP
# ==========================================================

start_time_sim = data.time
last_sim_time = data.time
last_print = 0.0
last_mode_print = ""

# Telemetry
telemetry_data = collections.defaultdict(list)
telemetry_headers = [
    "time", "state", "mode", "torso_x", "torso_y", "torso_z", 
    "torso_vx", "torso_vy", "torso_vz",
    "torso_roll", "torso_pitch", "torso_yaw",
    "torso_wx", "torso_wy", "torso_wz"
]
for jn in joint_names:
    telemetry_headers.extend([f"{jn}_pos_rad", f"{jn}_vel_rpm", f"{jn}_torque_nm"])

step_counter = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
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

        # --- UPDATE KEYBOARD (every 10ms) ---
        if step_counter % 10 == 0:
            update_keys()

        # --- FALL DETECTION ---
        if state >= 2 and state != 99 and state != 90 and state != 91 and is_fallen():
            print(f"  *** FALLEN at t={elapsed:.1f}s in state {state} mode={walk_mode} ***")
            state = 99

        # --- IDLE STATE (special handling, bypasses normal transition) ---
        if state == 10:
            if key_up:
                walk_mode = "forward"
                state = 2
            elif not HAS_KEYBOARD:
                # No keyboard library: auto-walk forward after 1s idle
                if state_elapsed > 1.0:
                    walk_mode = "forward"
                    state = 2

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
                else: state = 90  # Settle and stop
            elif state == 6: state = 7
            elif state == 7:
                state = 80
                step_count += 1
            elif state == 80: state = 81
            elif state == 81:
                # Decision point: continue walking or stop?
                if walk_mode == "forward" and key_up: state = 3
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
        for actuator_id, joint_name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]

            current_pos = data.qpos[qpos_addr]
            current_vel = data.qvel[qvel_addr]
            target_pos = (1.0 - alpha) * p1[joint_name] + alpha * p2[joint_name]

            kp, kd = get_gains(joint_name)
            torque = kp * (target_pos - current_pos) - kd * current_vel
            gear = model.actuator_gear[actuator_id, 0]
            data.ctrl[actuator_id] = np.clip(torque / gear, -1.0, 1.0)

        mujoco.mj_step(model, data)
        viewer.sync()

        # --- DIAGNOSTICS ---
        if elapsed - last_print > 1.0 and state >= 2 and state != 99 and state != 10:
            roll, pitch, _ = get_imu_data()
            mode_str = f"mode={walk_mode}"
            print(f"    [diag] t={elapsed:.1f}s st={state} {mode_str} roll={roll:.3f} pitch={pitch:.3f}")
            last_print = elapsed

        # --- TELEMETRY (100Hz) ---
        if step_counter % 10 == 0:
            roll, pitch, yaw = get_imu_data()
            torso_id = get_body_id("torso")
            vx, vy, vz = data.qvel[0:3]
            wx, wy, wz = data.qvel[3:6]
            row = [
                elapsed, state, walk_mode,
                data.xpos[torso_id, 0], data.xpos[torso_id, 1], data.xpos[torso_id, 2],
                vx, vy, vz,
                roll, pitch, yaw,
                wx, wy, wz
            ]
            for actuator_id, joint_name in enumerate(joint_names):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                pos = data.qpos[model.jnt_qposadr[jid]]
                vel = data.qvel[model.jnt_dofadr[jid]]
                rpm = vel * 60.0 / (2 * math.pi)
                gear = model.actuator_gear[actuator_id, 0]
                actual_torque = data.ctrl[actuator_id] * gear
                row.extend([pos, rpm, actual_torque])
            
            # Determine fine-grained action for telemetry grouping
            current_action = "idle"
            if walk_mode == "forward":
                current_action = "forward"

            telemetry_data[current_action].append(row)

        step_counter += 1

        remaining = (model.opt.timestep / PLAYBACK_SPEED) - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

print("Simulation finished. Writing telemetry files...")
for action, data_rows in telemetry_data.items():
    if not data_rows:
        continue
    filename = f"telemetry_{action}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(telemetry_headers)
        writer.writerows(data_rows)
    print(f"Saved {len(data_rows)} rows of telemetry data to {filename}.")
