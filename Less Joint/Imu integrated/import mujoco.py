import mujoco
import mujoco.viewer
import time
import math

model_path = "biped.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Anatomically Correct Poses ---
SIT_HIP, SIT_KNEE, SIT_ANKLE = 1.0, -2.0, 1.0

# STAND: Perfectly straight! Your mechanical CAD is naturally balanced.
STAND_HIP, STAND_KNEE, STAND_ANKLE = 0.0, 0.0, 0.0

# --- PD Balance Controller Tuning ---
Kp_balance = 1.0  
Kd_balance = 0.2  

# The Deadband Thresholds
DEADBAND_ANGLE = math.radians(0.5) 
DEADBAND_VEL = 0.05

# ====================================================================
def set_start_pose(joint_name, angle):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr] = angle

set_start_pose("right_hip", SIT_HIP)
set_start_pose("right_knee", SIT_KNEE)
set_start_pose("right_ankle", SIT_ANKLE)
set_start_pose("left_hip", SIT_HIP)
set_start_pose("left_knee", SIT_KNEE)
set_start_pose("left_ankle", SIT_ANKLE)

# Spawn perfectly on the floor
data.qpos[2] = 0.26  
# ====================================================================

r_hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_hip")
r_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_knee")
r_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_ankle")
l_hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_hip")
l_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_knee")
l_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_ankle")

sensor_tilt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_tilt")
sensor_gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")

print("Launching MuJoCo Viewer. Check this terminal to see live IMU Data!")
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        t = data.time
        
        # --- 1. Read the IMU ---
        tilt_adr = model.sensor_adr[sensor_tilt]
        raw_pitch = math.asin(max(-1.0, min(1.0, data.sensordata[tilt_adr])))
        
        gyro_adr = model.sensor_adr[sensor_gyro]
        raw_vel = data.sensordata[gyro_adr + 1]

        # --- THE SMOOTH DEADBAND FIX ---
        # Instead of snapping on and off, the error calculation eases in smoothly
        # once the robot crosses the threshold. No more "Bang-Bang" jitter!
        active_pitch = 0.0
        if raw_pitch > DEADBAND_ANGLE:
            active_pitch = raw_pitch - DEADBAND_ANGLE
        elif raw_pitch < -DEADBAND_ANGLE:
            active_pitch = raw_pitch + DEADBAND_ANGLE

        active_vel = 0.0
        if raw_vel > DEADBAND_VEL:
            active_vel = raw_vel - DEADBAND_VEL
        elif raw_vel < -DEADBAND_VEL:
            active_vel = raw_vel + DEADBAND_VEL

        # Calculate the ankle adjustment using the smoothed error
        balance_correction = (Kp_balance * active_pitch) + (Kd_balance * active_vel)

        if math.floor(t * 100) % 20 == 0:
            print(f"Time {t:.1f}s | Lean: {math.degrees(raw_pitch):.1f}° | Wobble: {raw_vel:.2f} | Push: {balance_correction:.3f}")

        # --- 2. Smooth "S-Curve" Stand Up ---
        balance_weight = max(0.0, min(1.0, t - 4.0))

        if t < 2.0:
            target_hip, target_knee, target_ankle = SIT_HIP, SIT_KNEE, SIT_ANKLE
            
        elif t < 5.0:
            p = (t - 2.0) / 3.0
            ease = (3 * p**2) - (2 * p**3) 
            target_hip = SIT_HIP + (STAND_HIP - SIT_HIP) * ease
            target_knee = SIT_KNEE + (STAND_KNEE - SIT_KNEE) * ease
            target_ankle = SIT_ANKLE + (STAND_ANKLE - SIT_ANKLE) * ease
            
        else:
            target_hip, target_knee, target_ankle = STAND_HIP, STAND_KNEE, STAND_ANKLE

        # --- 3. Apply the Balance Controller ---
        final_ankle = target_ankle + (balance_correction * balance_weight)

        data.ctrl[r_hip] = target_hip
        data.ctrl[r_knee] = target_knee
        data.ctrl[r_ankle] = final_ankle
        data.ctrl[l_hip] = target_hip
        data.ctrl[l_knee] = target_knee
        data.ctrl[l_ankle] = final_ankle

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)