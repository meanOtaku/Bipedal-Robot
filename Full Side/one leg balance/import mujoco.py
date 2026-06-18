import mujoco
import mujoco.viewer
import numpy as np
import time

# ==========================================================
# LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path("biped.xml")
data = mujoco.MjData(model)

print("Starting Sit-to-Stand-to-Balance Controller")

# ==========================================================
# JOINT LIST
# ==========================================================

joint_names = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",

    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll"
]

# ==========================================================
# POSTURES
# ==========================================================

sit_posture = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -1.0,
    "left_knee_pitch": 1.8,
    "left_ankle_pitch": -0.8,
    "left_ankle_roll": 0.0,

    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": -1.0,
    "right_knee_pitch": 1.8,
    "right_ankle_pitch": -0.8,
    "right_ankle_roll": 0.0
}

# The optimal lean angle to center the COM over the left foot
angle = 0.171

# Shift weight to the left leg while both feet are on the ground.
# We keep a slight bend in the knees (0.2) and hips (-0.1) so the legs 
# don't lock out completely, which prevents the robotic "snap" or jump.
shift_posture = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -angle,
    "left_hip_pitch": -0.1,
    "left_knee_pitch": 0.2,
    "left_ankle_pitch": -0.1,
    "left_ankle_roll": angle,

    "right_hip_yaw": 0.0,
    "right_hip_roll": -angle,
    "right_hip_pitch": -0.1,
    "right_knee_pitch": 0.2,
    "right_ankle_pitch": -0.1,
    "right_ankle_roll": angle
}

# Lift the right leg to balance on the left foot
stand_posture = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -angle,
    "left_hip_pitch": -0.1,
    "left_knee_pitch": 0.2,
    "left_ankle_pitch": -0.1,
    "left_ankle_roll": angle,

    "right_hip_yaw": 0.0,
    "right_hip_roll": -angle,
    "right_hip_pitch": -1.2,
    "right_knee_pitch": 1.8,
    "right_ankle_pitch": -0.5,
    "right_ankle_roll": angle
}

# ==========================================================
# INITIAL SITTING POSE
# ==========================================================

def apply_sitting_pose():

    # Root height
    data.qpos[2] = 0.90

    for joint_name in joint_names:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        qpos_addr = model.jnt_qposadr[jid]

        data.qpos[qpos_addr] = sit_posture[joint_name]

    mujoco.mj_forward(model, data)

apply_sitting_pose()

# ==========================================================
# PD GAINS
# ==========================================================

KP = {
    "yaw": 300,
    "roll": 500,
    "hip": 1200,
    "knee": 1500,
    "ankle": 800
}

# We increase KD (damping) slightly for hip, knee, and ankle
# to prevent the "springy" overshoot that causes the upward jump.
KD = {
    "yaw": 30,
    "roll": 50,
    "hip": 150,
    "knee": 180,
    "ankle": 120
}

def get_gains(name):

    if "yaw" in name:
        return KP["yaw"], KD["yaw"]

    if "roll" in name:
        return KP["roll"], KD["roll"]

    if "hip_pitch" in name:
        return KP["hip"], KD["hip"]

    if "knee" in name:
        return KP["knee"], KD["knee"]

    if "ankle" in name:
        return KP["ankle"], KD["ankle"]

    return 100, 10

# ==========================================================
# TIMING
# ==========================================================

sit_delay = 1.0
shift_duration = 2.0
lift_duration = 2.0

# Using data.time (Simulation Time) fixes the lagging bug!
start_time_sim = data.time

# Used to detect Reset button
last_sim_time = data.time

# ==========================================================
# VIEWER
# ==========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [0, 0, 0.8]
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -15

    while viewer.is_running():

        step_start = time.time()

        # ==================================================
        # DETECT RESET BUTTON
        # ==================================================

        if data.time < last_sim_time:

            print("Reset detected")

            apply_sitting_pose()

            start_time_sim = data.time

        last_sim_time = data.time

        # ==================================================
        # STAGED TRANSITION
        # ==================================================

        # Use deterministic simulation time to prevent inertial jolts
        elapsed = data.time - start_time_sim

        if elapsed < sit_delay:
            p1, p2, alpha = sit_posture, sit_posture, 0.0
        elif elapsed < sit_delay + shift_duration:
            p1, p2 = sit_posture, shift_posture
            alpha = (elapsed - sit_delay) / shift_duration
        elif elapsed < sit_delay + shift_duration + lift_duration:
            p1, p2 = shift_posture, stand_posture
            alpha = (elapsed - sit_delay - shift_duration) / lift_duration
        else:
            p1, p2, alpha = stand_posture, stand_posture, 1.0

        # Smooth interpolation
        alpha = 3 * alpha**2 - 2 * alpha**3

        # ==================================================
        # CONTROL
        # ==================================================

        for actuator_id, joint_name in enumerate(joint_names):

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )

            qpos_addr = model.jnt_qposadr[jid]
            qvel_addr = model.jnt_dofadr[jid]

            current_pos = data.qpos[qpos_addr]
            current_vel = data.qvel[qvel_addr]

            target_pos = (
                (1.0 - alpha) * p1[joint_name]
                + alpha * p2[joint_name]
            )

            kp, kd = get_gains(joint_name)

            torque = (
                kp * (target_pos - current_pos)
                - kd * current_vel
            )

            gear = model.actuator_gear[actuator_id, 0]

            data.ctrl[actuator_id] = np.clip(
                torque / gear,
                -1.0,
                1.0
            )

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = (
            model.opt.timestep
            - (time.time() - step_start)
        )

        if remaining > 0:
            time.sleep(remaining)

print("Simulation finished.")
