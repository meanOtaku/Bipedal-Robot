import mujoco
import mujoco.viewer
import numpy as np
import time

# ==========================================================
# LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path("biped.xml")
data = mujoco.MjData(model)

print("Starting Bipedal Walking Controller")

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

angle = 0.19  # Tuned for the new 0.15m hip separation

# ==========================================================
# KEYFRAMES (Fully Continuous Step-Through Gait)
# ==========================================================

sit = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -1.0, "left_knee_pitch": 1.8, "left_ankle_pitch": -0.8, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.0, "right_knee_pitch": 1.8, "right_ankle_pitch": -0.8, "right_ankle_roll": 0.0
}

stand = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0
}

# Initial shift to left leg (right leg is still adjacent)
lean_L_initial = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": angle
}

# Lift right leg (swing from -0.15 to -0.8)
lift_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.8, "right_knee_pitch": 1.2, "right_ankle_pitch": -0.4, "right_ankle_roll": angle
}

# Plant right foot (plant at -0.5)
plant_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": -0.5, "right_knee_pitch": 0.3, "right_ankle_pitch": 0.2, "right_ankle_roll": angle
}

# Shift weight right (Torso moves forward 0.35 rads)
# Right goes from -0.5 to -0.15. Left trails from -0.15 to +0.20 and STRAIGHTENS to maintain height.
center_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.20, "left_knee_pitch": 0.0, "left_ankle_pitch": -0.20, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0
}

lean_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": 0.20, "left_knee_pitch": 0.0, "left_ankle_pitch": -0.20, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

# Lift left leg (swing from +0.20 to -0.8)
lift_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.8, "left_knee_pitch": 1.2, "left_ankle_pitch": -0.4, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

# Plant left foot (plant at -0.5)
plant_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": angle, "left_hip_pitch": -0.5, "left_knee_pitch": 0.3, "left_ankle_pitch": 0.2, "left_ankle_roll": -angle,
    "right_hip_yaw": 0.0, "right_hip_roll": angle, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -angle
}

# Shift weight left (Torso moves forward 0.35 rads)
# Left goes from -0.5 to -0.15. Right trails from -0.15 to +0.20 and STRAIGHTENS to maintain height.
center_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.325, "left_knee_pitch": 0.3, "left_ankle_pitch": 0.025, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.025, "right_knee_pitch": 0.15, "right_ankle_pitch": -0.175, "right_ankle_roll": 0.0
}

lean_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -angle, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": angle,
    "right_hip_yaw": 0.0, "right_hip_roll": -angle, "right_hip_pitch": 0.20, "right_knee_pitch": 0.0, "right_ankle_pitch": -0.20, "right_ankle_roll": angle
}

# ==========================================================
# STATE MACHINE
# ==========================================================
# 0: SIT
# 1: STAND
# 2: SHIFT_INITIAL (stand -> lean_L_initial)
# 3: LIFT_RIGHT (lean_L/lean_L_initial -> lift_R)
# 4: PLANT_RIGHT (lift_R -> plant_R)
# 50: SHIFT_R_PHASE1 (plant_R -> center_R)
# 51: SHIFT_R_PHASE2 (center_R -> lean_R)
# 6: LIFT_LEFT (lean_R -> lift_L)
# 7: PLANT_LEFT (lift_L -> plant_L)
# 80: SHIFT_L_PHASE1 (plant_L -> center_L)
# 81: SHIFT_L_PHASE2 (center_L -> lean_L)

state = 0
state_start = 0.0
step_count = 0

MIN_DURATIONS = {
    0: 0.5,
    1: 1.0,    # stand
    2: 1.2,    # initial shift
    3: 0.3,    # lift right
    4: 0.2,    # plant right
    50: 0.8,   # shift R 1
    51: 0.8,   # shift R 2
    6: 0.3,    # lift left
    7: 0.2,    # plant left
    80: 0.8,   # shift L 1
    81: 0.8,   # shift L 2
    99: 999,   # fallen
}

def check_com_over_foot(side, threshold=0.04):
    com_y = get_com_y()
    foot_y = get_foot_y(side)
    return abs(com_y - foot_y) < threshold

def is_fallen():
    return get_torso_height() < 0.5

def get_poses_for_state(st):
    if st == 0: return sit, sit
    if st == 1: return sit, stand
    if st == 2: return stand, lean_L_initial
    if st == 3: return (lean_L_initial if step_count == 0 else lean_L), lift_R
    if st == 4: return lift_R, plant_R
    if st == 50: return plant_R, center_R
    if st == 51: return center_R, lean_R
    if st == 6: return lean_R, lift_L
    if st == 7: return lift_L, plant_L
    if st == 80: return plant_L, center_L
    if st == 81: return center_L, lean_L
    return stand, stand

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

KP = {"yaw": 600, "roll": 1500, "hip": 1200, "knee": 1500, "ankle_pitch": 1000, "ankle_roll": 1000}
KD = {"yaw": 60, "roll": 250, "hip": 150, "knee": 180, "ankle_pitch": 150, "ankle_roll": 150}

def get_gains(name):
    if "yaw" in name: return KP["yaw"], KD["yaw"]
    if "hip_roll" in name: return KP["roll"], KD["roll"]
    if "hip_pitch" in name: return KP["hip"], KD["hip"]
    if "knee" in name: return KP["knee"], KD["knee"]
    if "ankle_pitch" in name: return KP["ankle_pitch"], KD["ankle_pitch"]
    if "ankle_roll" in name: return KP["ankle_roll"], KD["ankle_roll"]
    return 100, 10

# ==========================================================
# SIMULATION LOOP
# ==========================================================

start_time_sim = data.time
last_sim_time = data.time
last_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15

    while viewer.is_running():
        step_start = time.time()

        if data.time < last_sim_time:
            print("Reset detected")
            apply_sitting_pose()
            start_time_sim = data.time
            state = 0
            state_start = 0.0
            step_count = 0
        last_sim_time = data.time

        elapsed = data.time - start_time_sim
        state_elapsed = elapsed - state_start
        min_dur = MIN_DURATIONS.get(state, 1.0)

        # Fall detection
        if state >= 2 and state != 99 and is_fallen():
            print(f"  *** FALLEN at t={elapsed:.1f}s in state {state} after {step_count} steps ***")
            state = 99

        # --- STATE TRANSITION ---
        can_transition = state_elapsed >= min_dur

        # For shift-to-foot states, also check COM
        if state == 2 and can_transition:
            if not check_com_over_foot('L', 0.05):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True
        elif state == 51 and can_transition:
            if not check_com_over_foot('R', 0.05):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True
        elif state == 81 and can_transition:
            if not check_com_over_foot('L', 0.05):
                can_transition = False
                if state_elapsed > min_dur + 3.0: can_transition = True

        if can_transition and state != 99:
            old_state = state
            if state == 0: state = 1
            elif state == 1: state = 2
            elif state == 2: state = 3
            elif state == 3: state = 4
            elif state == 4:
                state = 50
                step_count += 1
            elif state == 50: state = 51
            elif state == 51: state = 6
            elif state == 6: state = 7
            elif state == 7:
                state = 80
                step_count += 1
            elif state == 80: state = 81
            elif state == 81: state = 3

            state_start = elapsed
            if old_state != state:
                com_y = get_com_y()
                print(f"  t={elapsed:.1f}s  {old_state}->{state}  COM_y={com_y:.3f}  steps={step_count}")

        # --- COMPUTE TARGET ---
        if state == 99:
            p1, p2 = stand, stand
            alpha = 1.0
        else:
            p1, p2 = get_poses_for_state(state)
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

        # Diagnostics
        if elapsed - last_print > 1.0 and state >= 2 and state != 99:
            com_y = get_com_y()
            h = get_torso_height()
            print(f"    [diag] t={elapsed:.1f}s st={state} COM_y={com_y:.3f} h={h:.2f}")
            last_print = elapsed

        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

print("Simulation finished.")
