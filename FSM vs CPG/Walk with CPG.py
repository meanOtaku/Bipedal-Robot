"""
Walk With CPG v2 — State Machine + CPG Swing Trajectory
========================================================
Architecture:
  - BALANCE: Exact state machine from import mujoco.py (lean, CoM gate, settle)
  - SWING LEG: CPG spline generates smooth trajectory (speed-scaled)
  - STANCE LEG: Exact poses from import mujoco.py
  - PD GAINS: Exact values from import mujoco.py
  - ACTIVE BALANCE: Subtle ankle corrections from balance_test v5

Keyboard: UP arrow = walk forward, release = stop
Speed: Pass as command line arg (default 1.0, supports 1.0/1.4/1.8)
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import csv
import sys

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

if len(sys.argv) > 1:
    SPEED = float(sys.argv[1])
else:
    try:
        user_input = input("Enter walking speed (e.g., 1.0, 1.4, 1.8): ").strip()
        SPEED = float(user_input) if user_input else 1.0
    except ValueError:
        print("Invalid input. Defaulting to 1.0 m/s")
        SPEED = 1.0

# Automatically name the output CSV based on the speed
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else f"telemetry_cpg_{SPEED}.csv"

# ==========================================
# 1. CPG SWING TRAJECTORY
# ==========================================
# The CPG generates the swing leg's hip and knee trajectory as a function
# of phase [0.0 to 1.0]. Phase 0 = toe-off, Phase 0.5 = mid-swing (max lift),
# Phase 1.0 = heel-strike.
#
# Speed scaling: higher speed = larger hip swing + higher knee lift

def cpg_swing_trajectory(phase, speed):
    """
    Returns (hip_pitch, knee_pitch, ankle_pitch) for the SWING leg.
    Speed scaling: stride size grows gently, speed comes from faster timing.
    """
    # Gentle scaling: stride size grows slowly, speed mostly comes from timing
    # 1.0 -> 1.0, 1.4 -> 1.12, 1.8 -> 1.24
    stride_scale = 1.0 + 0.3 * (speed - 1.0)
    
    # Hip pitch: swings from back position to front position
    # Phase 0.0 = stance position (-0.15)
    # Phase 0.3 = maximum backward (-0.8 at speed 1.0) — leg swinging back/up
    # Phase 1.0 = forward plant (-0.5 at speed 1.0)
    hip_back = -0.15 - 0.65 * stride_scale   # -0.8 at speed 1.0
    hip_front = -0.15 - 0.35 * stride_scale  # -0.5 at speed 1.0
    
    if phase < 0.5:
        # First half: go from stance to max backward/lifted
        t = phase / 0.5
        t = 3*t**2 - 2*t**3  # smooth
        hip = -0.15 + (hip_back - (-0.15)) * t
    else:
        # Second half: swing forward to plant position
        t = (phase - 0.5) / 0.5
        t = 3*t**2 - 2*t**3
        hip = hip_back + (hip_front - hip_back) * t
    
    # Knee pitch: lifts high in first half, extends in second half
    knee_peak = 0.3 + 0.9 * stride_scale  # 1.2 at speed 1.0
    knee_plant = 0.3  # Always land with slightly bent knee
    
    if phase < 0.5:
        t = phase / 0.5
        t = 3*t**2 - 2*t**3
        knee = 0.3 + (knee_peak - 0.3) * t
    else:
        t = (phase - 0.5) / 0.5
        t = 3*t**2 - 2*t**3
        knee = knee_peak + (knee_plant - knee_peak) * t
    
    # Ankle: keep foot parallel to ground during swing
    ankle = -(hip + knee) * 0.5  # Rough compensation
    # Clamp ankle
    ankle = max(-0.5, min(0.3, ankle))
    
    # At phase 0.3 (peak lift), match import mujoco.py: ankle=-0.4
    # At phase 1.0 (plant), match import mujoco.py: ankle=0.2
    if phase < 0.5:
        t = phase / 0.5
        t = 3*t**2 - 2*t**3
        ankle = -0.15 + (-0.4 * stride_scale - (-0.15)) * t
    else:
        t = (phase - 0.5) / 0.5
        t = 3*t**2 - 2*t**3
        ankle = -0.4 * stride_scale + (0.2 * stride_scale - (-0.4 * stride_scale)) * t
    
    return hip, knee, ankle

# ==========================================
# 2. MUJOCO SETUP
# ==========================================
model = mujoco.MjModel.from_xml_path("biped.xml")
data = mujoco.MjData(model)

joint_names = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
]

# ==========================================
# 3. EXACT POSES FROM import mujoco.py
# ==========================================
angle = 0.19

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

# center_R: after planting right foot, shift weight to center
center_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.025, "left_knee_pitch": 0.15, "left_ankle_pitch": -0.175, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.325, "right_knee_pitch": 0.3, "right_ankle_pitch": 0.025, "right_ankle_roll": 0.0
}

lean_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.20, "left_knee_pitch": 0.0, "left_ankle_pitch": -0.20, "left_ankle_roll": -angle,
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

# ==========================================
# 4. EXACT PD GAINS FROM import mujoco.py
# ==========================================
KP = {"yaw": 1000, "roll": 1500, "hip": 1200, "knee": 1500, "ankle_pitch": 1000, "ankle_roll": 1000}
KD = {"yaw": 60, "roll": 250, "hip": 150, "knee": 180, "ankle_pitch": 150, "ankle_roll": 150}

def get_gains(name, state):
    if "yaw" in name: return 50.0, 5.0
    if "hip_roll" in name: return KP["roll"], KD["roll"]
    if "hip_pitch" in name: return KP["hip"], KD["hip"]
    if "knee" in name: return KP["knee"], KD["knee"]
    if "ankle_pitch" in name:
        # Dynamic Compliance: Relax trailing ankle during forward push
        if "left" in name and state in [50, 51, 6]:
            return 100, 10
        if "right" in name and state in [80, 81, 3]:
            return 100, 10
        return KP["ankle_pitch"], KD["ankle_pitch"]
    if "ankle_roll" in name: return KP["ankle_roll"], KD["ankle_roll"]
    return 100, 10

# ==========================================
# 5. HELPERS (exact from import mujoco.py)
# ==========================================
def get_body_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def check_com_over_foot(side, threshold=0.05):
    com_y = data.subtree_com[get_body_id("torso"), 1]
    if side == 'L':
        foot_y = data.geom("left_foot").xpos[1]
        return (com_y - foot_y) > -threshold
    else:
        foot_y = data.geom("right_foot").xpos[1]
        return (com_y - foot_y) < threshold

def quat_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    return roll, pitch

def get_imu_data():
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    adr = model.sensor_adr[sensor_id]
    quat = data.sensordata[adr:adr+4]
    return quat_to_euler(quat)

def is_fallen():
    roll, pitch = get_imu_data()
    return abs(roll) > 0.8 or abs(pitch) > 0.8

# ==========================================
# 6. INITIAL POSE
# ==========================================
data.qpos[2] = 0.90
for jn in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    data.qpos[model.jnt_qposadr[jid]] = sit[jn]
mujoco.mj_forward(model, data)

# ==========================================
# 7. STATE MACHINE (same as import mujoco.py)
# ==========================================
# 0:  SIT
# 1:  STAND
# 10: IDLE
# 2:  LEAN_L_INITIAL (stand -> lean_L_initial)
# 3:  SWING_R (CPG drives right leg trajectory)
# 50: SHIFT_R_1 (plant_R -> center_R)
# 51: SHIFT_R_2 (center_R -> lean_R)
# 6:  SWING_L (CPG drives left leg trajectory)
# 80: SHIFT_L_1 (plant_L -> center_L)
# 81: SHIFT_L_2 (center_L -> lean_L)
# 90: SETTLE_FROM_R
# 91: SETTLE_FROM_L

state = 0
state_start = 0.0
step_count = 0
walk_mode = "idle"

# Speed-scaled durations
speed_factor = SPEED / 1.0
swing_duration = max(0.3, 0.5 / speed_factor)  # Time for the CPG swing phase
shift_duration = 0.3  # Always give 0.3s for weight transfer (proven stable)

MIN_DURATIONS = {
    0: 0.5,
    1: 1.0,
    10: 999,
    2: 1.2,
    3: swing_duration,   # CPG swing right
    50: shift_duration,  # shift to center
    51: shift_duration,  # shift to lean_R
    6: swing_duration,   # CPG swing left
    80: shift_duration,  # shift to center
    81: shift_duration,  # shift to lean_L
    90: 1.0,
    91: 1.0,
    99: 999,
}

print(f"[INFO] CPG Walking Controller. Speed={SPEED} m/s")
print(f"  swing_dur={swing_duration:.2f}s, shift_dur={shift_duration:.2f}s")
print("=" * 50)
if HAS_KEYBOARD:
    print("  UP ARROW = Walk Forward")
    print("  Release  = Stop")
else:
    print("  Auto-walking (no keyboard)")
print("=" * 50)

# ==========================================
# 8. SIMULATION LOOP
# ==========================================
telemetry_data = []
telemetry_headers = [
    "time", "state", "mode", "torso_x", "torso_y", "torso_z", 
    "torso_vx", "torso_vy", "torso_vz",
    "torso_roll", "torso_pitch", "torso_yaw",
    "torso_wx", "torso_wy", "torso_wz"
]
for jn in joint_names:
    telemetry_headers.extend([f"{jn}_pos_rad", f"{jn}_vel_rpm", f"{jn}_torque_nm"])

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15

    start_time_sim = data.time
    last_sim_time = data.time
    old_state = -1
    current_p1 = sit
    current_p2 = sit
    step_counter = 0
    elapsed = 0.0

    while viewer.is_running() and elapsed < 125.0:
        step_start = time.time()
        
        # Time tracking
        if data.time < last_sim_time:
            # Reset
            data.qpos[2] = 0.90
            for jn in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                data.qpos[model.jnt_qposadr[jid]] = sit[jn]
            mujoco.mj_forward(model, data)
            start_time_sim = data.time
            state = 0
            state_start = 0.0
            step_count = 0
            walk_mode = "idle"
            current_p1 = sit
            current_p2 = sit
            old_state = -1
        last_sim_time = data.time
        
        elapsed = data.time - start_time_sim
        state_elapsed = elapsed - state_start
        min_dur = MIN_DURATIONS.get(state, 1.0)
        
        # Keyboard
        if step_counter % 10 == 0 and HAS_KEYBOARD:
            try:
                key_up = keyboard.is_pressed('up')
            except:
                key_up = False
        elif not HAS_KEYBOARD:
            key_up = True  # Auto-walk
        
        # Fall detection
        if state >= 2 and state != 99 and state != 90 and state != 91 and is_fallen():
            print(f"  *** FALLEN at t={elapsed:.1f}s in state {state} ***")
            state = 99
        
        # IDLE handling
        if state == 10:
            if key_up:
                walk_mode = "forward"
                state = 2
            elif not HAS_KEYBOARD and elapsed > 2.0:
                walk_mode = "forward"
                state = 2
        
        # ==========================================
        # STATE TRANSITIONS
        # ==========================================
        can_transition = state_elapsed >= min_dur
        
        # CoM gates (exact same as import mujoco.py)
        com_threshold = 0.05
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
            elif state == 3:
                state = 50
                step_count += 1
            elif state == 50: state = 51
            elif state == 51:
                if walk_mode == "forward" and key_up:
                    state = 6
                else:
                    state = 90
            elif state == 6:
                state = 80
                step_count += 1
            elif state == 80: state = 81
            elif state == 81:
                if walk_mode == "forward" and key_up:
                    state = 3
                else:
                    state = 91
            elif state == 90:
                state = 10
                walk_mode = "idle"
                step_count = 0
            elif state == 91:
                state = 10
                walk_mode = "idle"
                step_count = 0
        
        # ==========================================
        # POSE COMPUTATION
        # ==========================================
        if state != old_state:
            state_start = elapsed
            
            # When leaving a CPG swing state, build the correct end pose
            if old_state == 3:
                # End of SWING_R: build plant_R pose from CPG at phase=1.0
                sh, sk, sa = cpg_swing_trajectory(1.0, SPEED)
                current_p1 = {
                    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15,
                    "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
                    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": sh,
                    "right_knee_pitch": sk, "right_ankle_pitch": sa, "right_ankle_roll": angle
                }
            elif old_state == 6:
                # End of SWING_L: build plant_L pose from CPG at phase=1.0
                sh, sk, sa = cpg_swing_trajectory(1.0, SPEED)
                current_p1 = {
                    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": sh,
                    "left_knee_pitch": sk, "left_ankle_pitch": sa, "left_ankle_roll": -angle,
                    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15,
                    "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
                }
            else:
                current_p1 = dict(current_p2)
            
            if state == 0: current_p1, current_p2 = sit, sit
            elif state == 1: current_p1 = sit; current_p2 = stand
            elif state == 10: current_p2 = stand
            elif state == 2: current_p2 = lean_L_initial
            # States 3, 6: CPG handles these (computed below)
            elif state == 50: current_p2 = center_R
            elif state == 51: current_p2 = lean_R
            elif state == 80: current_p2 = center_L
            elif state == 81: current_p2 = lean_L
            elif state == 90: current_p2 = stand
            elif state == 91: current_p2 = stand
            
            if state not in [3, 6, 10, 99]:
                state_names = {0: "SIT", 1: "STAND", 2: "LEAN_L", 50: "CENTER_R", 51: "LEAN_R", 
                              80: "CENTER_L", 81: "LEAN_L", 90: "SETTLE_R", 91: "SETTLE_L"}
                if state in state_names:
                    print(f"  t={elapsed:.1f}s  -> {state_names[state]}  steps={step_count}")
            elif state == 3:
                print(f"  t={elapsed:.1f}s  -> SWING_R (CPG)  steps={step_count}")
            elif state == 6:
                print(f"  t={elapsed:.1f}s  -> SWING_L (CPG)  steps={step_count}")
            
            old_state = state
        
        # ==========================================
        # COMPUTE TARGET POSE
        # ==========================================
        target_pose = {}
        
        if state == 3:
            # ===== CPG SWING RIGHT =====
            # Phase goes from 0.0 to 1.0 over swing_duration
            phase = min(1.0, state_elapsed / swing_duration)
            swing_hip, swing_knee, swing_ankle = cpg_swing_trajectory(phase, SPEED)
            
            # Left leg (stance): hold lean_L pose
            target_pose["left_hip_yaw"] = 0.0
            target_pose["left_hip_roll"] = -angle
            target_pose["left_hip_pitch"] = -0.15
            target_pose["left_knee_pitch"] = 0.3
            target_pose["left_ankle_pitch"] = -0.15
            target_pose["left_ankle_roll"] = angle
            
            # Right leg (swing): CPG trajectory
            target_pose["right_hip_yaw"] = 0.0
            target_pose["right_hip_roll"] = -angle
            target_pose["right_hip_pitch"] = swing_hip
            target_pose["right_knee_pitch"] = swing_knee
            target_pose["right_ankle_pitch"] = swing_ankle
            target_pose["right_ankle_roll"] = angle
            
        elif state == 6:
            # ===== CPG SWING LEFT =====
            phase = min(1.0, state_elapsed / swing_duration)
            swing_hip, swing_knee, swing_ankle = cpg_swing_trajectory(phase, SPEED)
            
            # Right leg (stance): hold lean_R pose
            target_pose["right_hip_yaw"] = 0.0
            target_pose["right_hip_roll"] = angle
            target_pose["right_hip_pitch"] = -0.15
            target_pose["right_knee_pitch"] = 0.3
            target_pose["right_ankle_pitch"] = -0.15
            target_pose["right_ankle_roll"] = -angle
            
            # Left leg (swing): CPG trajectory
            target_pose["left_hip_yaw"] = 0.0
            target_pose["left_hip_roll"] = angle
            target_pose["left_hip_pitch"] = swing_hip
            target_pose["left_knee_pitch"] = swing_knee
            target_pose["left_ankle_pitch"] = swing_ankle
            target_pose["left_ankle_roll"] = -angle
            
        else:
            # ===== STANDARD INTERPOLATION (all other states) =====
            alpha = float(np.clip(state_elapsed / min_dur, 0.0, 1.0))
            alpha = 3 * alpha**2 - 2 * alpha**3  # Smooth ease-in-out
            
            for jn in joint_names:
                target_pose[jn] = (1.0 - alpha) * current_p1.get(jn, 0.0) + alpha * current_p2.get(jn, 0.0)
        
        # ==========================================
        # ACTIVE ANKLE BALANCE (subtle, from balance_test v5)
        # ==========================================
        if state in [3, 6, 50, 51, 80, 81]:
            gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
            gyro_adr = model.sensor_adr[gyro_id]
            
            roll_imu, pitch_imu = get_imu_data()
            pitch_rate = data.sensordata[gyro_adr + 1]
            
            pitch_corr = -0.1 * pitch_imu - 0.02 * pitch_rate
            pitch_corr = np.clip(pitch_corr, -0.03, 0.03)
            
            # Apply to stance leg only
            if state in [3, 50, 51]:  # Left leg is stance
                target_pose["left_ankle_pitch"] += pitch_corr
            elif state in [6, 80, 81]:  # Right leg is stance
                target_pose["right_ankle_pitch"] += pitch_corr
        
        # ==========================================
        # PD CONTROL (exact from import mujoco.py)
        # ==========================================
        for actuator_id, joint_name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]

            current_pos = data.qpos[qpos_addr]
            current_vel = data.qvel[qvel_addr]
            target_pos = target_pose.get(joint_name, 0.0)

            kp, kd = get_gains(joint_name, state)
            torque = kp * (target_pos - current_pos) - kd * current_vel
            gear = model.actuator_gear[actuator_id, 0]
            data.ctrl[actuator_id] = np.clip(torque / gear, -1.0, 1.0)

        # Telemetry (100Hz)
        if step_counter % 10 == 0:
            x, y, z = data.qpos[0:3]
            vx, vy, vz = data.qvel[0:3]
            
            # Quat to euler for torso
            q = data.qpos[3:7]
            qw, qx, qy, qz = q
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (qw * qy - qz * qx)
            pitch = np.arcsin(np.clip(sinp, -1, 1))
            
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            wx, wy, wz = data.qvel[3:6]
            
            row = [elapsed, state, walk_mode, x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            for actuator_id, joint_name in enumerate(joint_names):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                pos = data.qpos[model.jnt_qposadr[jid]]
                vel = data.qvel[model.jnt_dofadr[jid]]
                rpm = vel * 60.0 / (2 * math.pi)
                actual_torque = data.ctrl[actuator_id] * model.actuator_gear[actuator_id, 0]
                row.extend([pos, rpm, actual_torque])
            telemetry_data.append(row)

        mujoco.mj_step(model, data)
        viewer.sync()
        step_counter += 1

        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

# Save telemetry
print(f"[SUCCESS] CPG Walking complete. Saving to {OUTPUT_CSV}")
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(telemetry_headers)
    writer.writerows(telemetry_data)
