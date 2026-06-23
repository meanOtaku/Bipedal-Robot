import time

import mujoco
import mujoco.viewer
import numpy as np

from .config import MODEL_PATH, PLAYBACK_SPEED, TRAJECTORY_FLUSH_EVERY_STEPS, TRAJECTORY_SAMPLE_EVERY_STEPS
from .gains import get_gains
from .input import ViewerCommand
from .kinematics import body_id, get_imu_data
from .model_constants import FOOT_GEOM_NAMES, JOINT_NAMES, TRAJECTORY_BODY_NAMES
from .trajectory_logger import TrajectoryLogger


ANGLE = 0.19

sit = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -1.0, "left_knee_pitch": 1.8, "left_ankle_pitch": -0.8, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.0, "right_knee_pitch": 1.8, "right_ankle_pitch": -0.8, "right_ankle_roll": 0.0,
}

stand = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0,
}

lean_L_initial = {
    "left_hip_yaw": 0.0, "left_hip_roll": -ANGLE, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": ANGLE,
    "right_hip_yaw": 0.0, "right_hip_roll": -ANGLE, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": ANGLE,
}

center_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.025, "left_knee_pitch": 0.15, "left_ankle_pitch": -0.175, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.325, "right_knee_pitch": 0.3, "right_ankle_pitch": 0.025, "right_ankle_roll": 0.0,
}

lean_R = {
    "left_hip_yaw": 0.0, "left_hip_roll": ANGLE, "left_hip_pitch": 0.20, "left_knee_pitch": 0.0, "left_ankle_pitch": -0.20, "left_ankle_roll": -ANGLE,
    "right_hip_yaw": 0.0, "right_hip_roll": ANGLE, "right_hip_pitch": -0.15, "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -ANGLE,
}

center_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.325, "left_knee_pitch": 0.3, "left_ankle_pitch": 0.025, "left_ankle_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.025, "right_knee_pitch": 0.15, "right_ankle_pitch": -0.175, "right_ankle_roll": 0.0,
}

lean_L = {
    "left_hip_yaw": 0.0, "left_hip_roll": -ANGLE, "left_hip_pitch": -0.15, "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": ANGLE,
    "right_hip_yaw": 0.0, "right_hip_roll": -ANGLE, "right_hip_pitch": 0.20, "right_knee_pitch": 0.30, "right_ankle_pitch": -0.20, "right_ankle_roll": ANGLE,
}


def smoothstep(t):
    return 3 * t**2 - 2 * t**3


def cpg_swing_trajectory(phase, speed):
    """Return hip, knee, and ankle pitch targets for the active swing leg."""
    phase = float(np.clip(phase, 0.0, 1.0))
    stride_scale = 1.0 + 0.3 * (speed - 1.0)
    hip_back = -0.15 - 0.65 * stride_scale
    hip_front = -0.15 - 0.35 * stride_scale

    if phase < 0.5:
        t = smoothstep(phase / 0.5)
        hip = -0.15 + (hip_back + 0.15) * t
        knee_peak = 0.3 + 0.9 * stride_scale
        knee = 0.3 + (knee_peak - 0.3) * t
        ankle = -0.15 + (-0.4 * stride_scale + 0.15) * t
    else:
        t = smoothstep((phase - 0.5) / 0.5)
        knee_peak = 0.3 + 0.9 * stride_scale
        hip = hip_back + (hip_front - hip_back) * t
        knee = knee_peak + (0.3 - knee_peak) * t
        ankle = -0.4 * stride_scale + (0.2 * stride_scale + 0.4 * stride_scale) * t

    return hip, knee, float(np.clip(ankle, -0.5, 0.3))


def build_right_plant_pose(speed):
    hip, knee, ankle = cpg_swing_trajectory(1.0, speed)
    return {
        "left_hip_yaw": 0.0, "left_hip_roll": -ANGLE, "left_hip_pitch": -0.15,
        "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": ANGLE,
        "right_hip_yaw": 0.0, "right_hip_roll": -ANGLE, "right_hip_pitch": hip,
        "right_knee_pitch": knee, "right_ankle_pitch": ankle, "right_ankle_roll": ANGLE,
    }


def build_left_plant_pose(speed):
    hip, knee, ankle = cpg_swing_trajectory(1.0, speed)
    return {
        "left_hip_yaw": 0.0, "left_hip_roll": ANGLE, "left_hip_pitch": hip,
        "left_knee_pitch": knee, "left_ankle_pitch": ankle, "left_ankle_roll": -ANGLE,
        "right_hip_yaw": 0.0, "right_hip_roll": ANGLE, "right_hip_pitch": -0.15,
        "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -ANGLE,
    }


def apply_sitting_pose(model, data):
    data.qpos[2] = 0.90
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = sit[joint_name]
    mujoco.mj_forward(model, data)


def check_com_over_foot(model, data, side, threshold=0.05):
    com_y = data.subtree_com[body_id(model, "torso"), 1]
    foot_name = "left_foot" if side == "L" else "right_foot"
    foot_y = data.geom(foot_name).xpos[1]
    if side == "L":
        return com_y - foot_y > -threshold
    return com_y - foot_y < threshold


def is_fallen(model, data):
    roll, pitch, _ = get_imu_data(model, data)
    return abs(roll) > 0.8 or abs(pitch) > 0.8


def run(speed=1.0):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    viewer_command = ViewerCommand()
    apply_sitting_pose(model, data)

    speed = max(0.1, float(speed))
    swing_duration = max(0.3, 0.5 / speed)
    shift_duration = 0.3
    min_durations = {
        0: 0.5,
        1: 1.0,
        10: 999,
        2: 1.2,
        3: swing_duration,
        50: shift_duration,
        51: shift_duration,
        6: swing_duration,
        80: shift_duration,
        81: shift_duration,
        90: 1.0,
        91: 1.0,
        99: 999,
    }

    print("Starting Bipedal Walking Controller")
    print("=" * 50)
    print(f"  GAIT MODE: CPG  speed={speed:.2f}")
    print("  VIEWER CONTROLS:")
    print("    W / Up Arrow       -  Walk Forward")
    print("    Space / Esc        -  Stop")
    print("=" * 50)

    state = 0
    state_start = 0.0
    step_count = 0
    walk_mode = "idle"
    start_time_sim = data.time
    last_sim_time = data.time
    last_print = 0.0
    old_state = -1
    current_p1 = sit
    current_p2 = sit
    step_counter = 0
    trajectory_logger = TrajectoryLogger(model, data, JOINT_NAMES, TRAJECTORY_BODY_NAMES, FOOT_GEOM_NAMES)

    with mujoco.viewer.launch_passive(model, data, key_callback=viewer_command.handle_key) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15

        while viewer.is_running():
            step_start = time.time()

            if data.time < last_sim_time:
                print("Reset detected")
                apply_sitting_pose(model, data)
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
            min_dur = min_durations.get(state, 1.0)
            key_up = viewer_command.key_up

            if state >= 2 and state not in [90, 91, 99] and is_fallen(model, data):
                print(f"  *** FALLEN at t={elapsed:.1f}s in state {state} mode={walk_mode} ***")
                state = 99

            if state == 10:
                if key_up:
                    walk_mode = "forward"
                    state = 2
                elif viewer_command.motion_command == "idle":
                    walk_mode = "idle"

            can_transition = state_elapsed >= min_dur
            if state == 2 and can_transition:
                if not check_com_over_foot(model, data, "L"):
                    can_transition = state_elapsed > min_dur + 3.0
            elif state == 51 and can_transition:
                if not check_com_over_foot(model, data, "R"):
                    can_transition = state_elapsed > min_dur + 3.0
            elif state == 81 and can_transition:
                if not check_com_over_foot(model, data, "L"):
                    can_transition = state_elapsed > min_dur + 3.0

            if can_transition and state not in [10, 99]:
                if state == 0:
                    state = 1
                elif state == 1:
                    state = 10
                    walk_mode = "idle"
                elif state == 2:
                    state = 3
                elif state == 3:
                    state = 50
                    step_count += 1
                elif state == 50:
                    state = 51
                elif state == 51:
                    state = 6 if walk_mode == "forward" and key_up else 90
                elif state == 6:
                    state = 80
                    step_count += 1
                elif state == 80:
                    state = 81
                elif state == 81:
                    state = 3 if walk_mode == "forward" and key_up else 91
                elif state in [90, 91]:
                    state = 10
                    walk_mode = "idle"
                    step_count = 0

            if state != old_state:
                state_start = elapsed
                if old_state == 3:
                    current_p1 = build_right_plant_pose(speed)
                elif old_state == 6:
                    current_p1 = build_left_plant_pose(speed)
                else:
                    current_p1 = dict(current_p2)

                if state == 0:
                    current_p1, current_p2 = sit, sit
                elif state == 1:
                    current_p1, current_p2 = sit, stand
                elif state == 10:
                    current_p2 = stand
                elif state == 2:
                    current_p2 = lean_L_initial
                elif state == 50:
                    current_p2 = center_R
                elif state == 51:
                    current_p2 = lean_R
                elif state == 80:
                    current_p2 = center_L
                elif state == 81:
                    current_p2 = lean_L
                elif state in [90, 91]:
                    current_p2 = stand

                if state == 3:
                    print(f"  t={elapsed:.1f}s  -> SWING_R (CPG)  steps={step_count}")
                elif state == 6:
                    print(f"  t={elapsed:.1f}s  -> SWING_L (CPG)  steps={step_count}")
                elif state == 2:
                    print(f"  t={elapsed:.1f}s  IDLE->WALK_CPG")
                elif state in [90, 91]:
                    print(f"  t={elapsed:.1f}s  SETTLING")
                elif state == 10 and old_state != -1 and old_state != 1:
                    print(f"  t={elapsed:.1f}s  IDLE (stopped walking)")
                old_state = state
                state_elapsed = 0.0
                min_dur = min_durations.get(state, 1.0)

            target_joint_positions = {}
            if state == 3:
                phase = min(1.0, state_elapsed / swing_duration)
                swing_hip, swing_knee, swing_ankle = cpg_swing_trajectory(phase, speed)
                target_joint_positions = {
                    "left_hip_yaw": 0.0, "left_hip_roll": -ANGLE, "left_hip_pitch": -0.15,
                    "left_knee_pitch": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": ANGLE,
                    "right_hip_yaw": 0.0, "right_hip_roll": -ANGLE, "right_hip_pitch": swing_hip,
                    "right_knee_pitch": swing_knee, "right_ankle_pitch": swing_ankle, "right_ankle_roll": ANGLE,
                }
            elif state == 6:
                phase = min(1.0, state_elapsed / swing_duration)
                swing_hip, swing_knee, swing_ankle = cpg_swing_trajectory(phase, speed)
                target_joint_positions = {
                    "left_hip_yaw": 0.0, "left_hip_roll": ANGLE, "left_hip_pitch": swing_hip,
                    "left_knee_pitch": swing_knee, "left_ankle_pitch": swing_ankle, "left_ankle_roll": -ANGLE,
                    "right_hip_yaw": 0.0, "right_hip_roll": ANGLE, "right_hip_pitch": -0.15,
                    "right_knee_pitch": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": -ANGLE,
                }
            elif state == 99:
                target_joint_positions = dict(stand)
            else:
                alpha = float(np.clip(state_elapsed / min_dur, 0.0, 1.0))
                alpha = smoothstep(alpha)
                for joint_name in JOINT_NAMES:
                    target_joint_positions[joint_name] = (
                        (1.0 - alpha) * current_p1.get(joint_name, 0.0)
                        + alpha * current_p2.get(joint_name, 0.0)
                    )

            if state in [3, 6, 50, 51, 80, 81]:
                gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
                gyro_adr = model.sensor_adr[gyro_id]
                _, pitch_imu, _ = get_imu_data(model, data)
                pitch_rate = data.sensordata[gyro_adr + 1]
                pitch_corr = float(np.clip(-0.1 * pitch_imu - 0.02 * pitch_rate, -0.03, 0.03))
                if state in [3, 50, 51]:
                    target_joint_positions["left_ankle_pitch"] += pitch_corr
                elif state in [6, 80, 81]:
                    target_joint_positions["right_ankle_pitch"] += pitch_corr

            for actuator_id, joint_name in enumerate(JOINT_NAMES):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_addr = model.jnt_qposadr[joint_id]
                qvel_addr = model.jnt_dofadr[joint_id]
                current_pos = data.qpos[qpos_addr]
                current_vel = data.qvel[qvel_addr]
                target_pos = target_joint_positions[joint_name]
                kp, kd = get_gains(joint_name, "forward", state)
                torque = kp * (target_pos - current_pos) - kd * current_vel
                gear = model.actuator_gear[actuator_id, 0]
                data.ctrl[actuator_id] = np.clip(torque / gear, -1.0, 1.0)

            mujoco.mj_step(model, data)
            viewer.sync()

            if elapsed - last_print > 1.0 and state >= 2 and state not in [10, 99]:
                roll, pitch, _ = get_imu_data(model, data)
                print(f"    [diag] t={elapsed:.1f}s st={state} mode=cpg roll={roll:.3f} pitch={pitch:.3f}")
                last_print = elapsed

            if step_counter % TRAJECTORY_SAMPLE_EVERY_STEPS == 0:
                current_action = "cpg_forward" if walk_mode == "forward" else "idle"
                trajectory_logger.record(
                    elapsed,
                    state,
                    "cpg_forward" if walk_mode == "forward" else walk_mode,
                    viewer_command.motion_command,
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
