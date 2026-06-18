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
# JOINT LIST
# ==========================================================

joint_names = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
]

# ==========================================================
# GAIT PARAMETERS & KEYFRAMES
# ==========================================================

# Slightly reduced lean angle to prevent overbalancing sideways during a dynamic step
walk_angle = 0.13 

# Stride parameters
stride_p = 0.2       # Shorter, more stable steps
knee_stance = 0.1    # Knee bend during stance phase
knee_swing = 0.8     # Knee bend during swing phase (increased to prevent dragging)
toe_clearance = 0.2  # Ankle adjustment during swing to prevent toe stubbing

def make_pose(lean, l_pitch, l_knee, r_pitch, r_knee, l_toe_up=0.0, r_toe_up=0.0):
    """
    Helper function to generate a full-body posture dict.
    Calculates ankle pitches automatically to keep the feet flat on the floor.
    """
    if lean == 'L':
        roll = -walk_angle
        a_roll = walk_angle
    elif lean == 'R':
        roll = walk_angle
        a_roll = -walk_angle
    else:
        roll = 0.0
        a_roll = 0.0
        
    # Apply an offset to the hip pitch based on the knee bend. 
    # Without this, bending the knee shifts the foot backward and the robot falls over!
    l_hip = l_pitch - l_knee / 2.0
    r_hip = r_pitch - r_knee / 2.0

    # Ankle pitch compensates for hip and knee pitch to keep the foot flat
    l_ankle = -l_hip - l_knee + l_toe_up
    r_ankle = -r_hip - r_knee + r_toe_up
        
    return {
        "left_hip_yaw": 0.0, "left_hip_roll": roll, "left_hip_pitch": l_hip, "left_knee_pitch": l_knee, "left_ankle_pitch": l_ankle, "left_ankle_roll": a_roll,
        "right_hip_yaw": 0.0, "right_hip_roll": roll, "right_hip_pitch": r_hip, "right_knee_pitch": r_knee, "right_ankle_pitch": r_ankle, "right_ankle_roll": a_roll
    }

# Start sitting, then stand up
sit_posture = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -1.0, "left_knee_pitch": 1.8, "left_ankle_pitch": -0.8, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.0, "right_knee_pitch": 1.8, "right_ankle_pitch": -0.8, "right_ankle_roll": 0.0
}
stand_both = make_pose('C', 0.0, knee_stance, 0.0, knee_stance)

# Startup Walk
p_lean_L_start = make_pose('L', 0.0, knee_stance, 0.0, knee_stance)
p_swing_R_start = make_pose('L', stride_p, knee_stance, -stride_p, knee_swing, r_toe_up=toe_clearance)
p_drop_R_start = make_pose('L', stride_p, knee_stance, -stride_p, knee_stance)

# Walking cycle keyframes
# Shift weight to Right (Front foot). Torso moves forward over Right foot.
p_lean_R = make_pose('R', 2*stride_p, knee_stance, 0.0, knee_stance)
# Swing Left leg forward. Right leg (stance) moves backward relative to torso.
p_swing_L = make_pose('R', -stride_p, knee_swing, stride_p, knee_stance, l_toe_up=toe_clearance)
# Drop Left foot
p_drop_L = make_pose('R', -stride_p, knee_stance, stride_p, knee_stance)

# Shift weight to Left (Front foot). Torso moves forward over Left foot.
p_lean_L = make_pose('L', 0.0, knee_stance, 2*stride_p, knee_stance)
# Swing Right leg forward. Left leg (stance) moves backward relative to torso.
p_swing_R = make_pose('L', stride_p, knee_stance, -stride_p, knee_swing, r_toe_up=toe_clearance)
# Drop Right foot
p_drop_R = make_pose('L', stride_p, knee_stance, -stride_p, knee_stance)

# Timing lists: (target_posture, duration_in_seconds)
startup_sequence = [
    (sit_posture, 0.5),       # Sit briefly
    (stand_both, 1.0),        # Stand up faster
    (p_lean_L_start, 0.5),    # Shift weight left
    (p_lean_L_start, 0.2),    # Hold to let lateral momentum settle
    (p_swing_R_start, 0.4),   # Quick swing right
    (p_drop_R_start, 0.15)    # Plant right foot
]

cycle_sequence = [
    (p_lean_R, 1.0),          # Takes 1.0s to shift from Lean L to Lean R (matches 0.5s from Center to L)
    (p_lean_R, 0.2),          # Hold to let lateral momentum settle
    (p_swing_L, 0.6),         # Slightly longer to accommodate the full swing arc
    (p_drop_L, 0.2),          # Drop left foot
    (p_lean_L, 1.0),          # Takes 1.0s to shift from Lean R to Lean L
    (p_lean_L, 0.2),          # Hold to let lateral momentum settle
    (p_swing_R, 0.6),         # Swing right leg
    (p_drop_R, 0.2)           # Drop right foot
]

# ==========================================================
# INITIAL SITTING POSE
# ==========================================================

def apply_sitting_pose():
    data.qpos[2] = 0.90 # Root height

    for joint_name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = model.jnt_qposadr[jid]
        data.qpos[qpos_addr] = sit_posture[joint_name]

    mujoco.mj_forward(model, data)

apply_sitting_pose()

# ==========================================================
# PD GAINS
# ==========================================================

KP = {"yaw": 300, "roll": 500, "hip": 1200, "knee": 1500, "ankle": 800}
KD = {"yaw": 30, "roll": 150, "hip": 150, "knee": 180, "ankle": 120}

def get_gains(name):
    if "yaw" in name: return KP["yaw"], KD["yaw"]
    if "roll" in name: return KP["roll"], KD["roll"]
    if "hip_pitch" in name: return KP["hip"], KD["hip"]
    if "knee" in name: return KP["knee"], KD["knee"]
    if "ankle" in name: return KP["ankle"], KD["ankle"]
    return 100, 10

# ==========================================================
# VIEWER & SIMULATION LOOP
# ==========================================================

start_time_sim = data.time
last_sim_time = data.time

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -15

    while viewer.is_running():
        step_start = time.time()

        # DETECT RESET BUTTON
        if data.time < last_sim_time:
            print("Reset detected")
            apply_sitting_pose()
            start_time_sim = data.time

        last_sim_time = data.time

        # ==================================================
        # SEQUENCE LOGIC
        # ==================================================
        elapsed = data.time - start_time_sim
        startup_time = sum(dur for pos, dur in startup_sequence)

        if elapsed < startup_time:
            current_t = 0.0
            for i, (pos, dur) in enumerate(startup_sequence):
                if elapsed <= current_t + dur:
                    p2 = pos
                    p1 = startup_sequence[i-1][0] if i > 0 else sit_posture
                    alpha = (elapsed - current_t) / dur
                    break
                current_t += dur
        else:
            cycle_time = sum(dur for pos, dur in cycle_sequence)
            t_in_cycle = (elapsed - startup_time) % cycle_time
            
            current_t = 0.0
            for i, (pos, dur) in enumerate(cycle_sequence):
                if t_in_cycle <= current_t + dur:
                    p2 = pos
                    if i == 0:
                        # Smooth transition from startup to cycle, or loop back around
                        p1 = startup_sequence[-1][0] if (elapsed - startup_time) < cycle_time else cycle_sequence[-1][0]
                    else:
                        p1 = cycle_sequence[i-1][0]
                    alpha = (t_in_cycle - current_t) / dur
                    break
                current_t += dur

        # Smooth interpolation
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = 3 * alpha**2 - 2 * alpha**3

        # ==================================================
        # CONTROL
        # ==================================================
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

        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

print("Simulation finished.")
