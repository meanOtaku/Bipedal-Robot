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

# --- Anatomically Correct Poses (in Radians) ---
# Hip bends forward (+), Knee bends backward (-), Ankle bends forward (+) to stay flat
SIT_HIP, SIT_KNEE, SIT_ANKLE = 1.0, -2.0, 1.0
STAND_HIP, STAND_KNEE, STAND_ANKLE = 0.0, 0.0, 0.0

# ====================================================================
# CRITICAL FIX: Pre-Bend the physics geometry BEFORE simulation starts
# ====================================================================
def set_start_pose(joint_name, angle):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_adr = model.jnt_qposadr[jnt_id]
    data.qpos[qpos_adr] = angle

# Fold the legs into the 'Z' sitting shape
set_start_pose("right_hip", SIT_HIP)
set_start_pose("right_knee", SIT_KNEE)
set_start_pose("right_ankle", SIT_ANKLE)
set_start_pose("left_hip", SIT_HIP)
set_start_pose("left_knee", SIT_KNEE)
set_start_pose("left_ankle", SIT_ANKLE)
# Drop the slide joint slightly so the bent legs rest perfectly on the floor
set_start_pose("root_z", -0.15)
# ====================================================================

# Fetch the Position Actuators
r_hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_hip")
r_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_knee")
r_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_right_ankle")
l_hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_hip")
l_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_knee")
l_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_left_ankle")

print("Launching MuJoCo Viewer. Press ESC or close the window to exit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        t = data.time
        
        # --- The Stand-Up Choreography ---
        if t < 2.0:
            # First 2 Seconds: Lock servos in the sitting position
            current_hip, current_knee, current_ankle = SIT_HIP, SIT_KNEE, SIT_ANKLE
            
        elif t < 5.0:
            # 2.0s to 5.0s: Smoothly transition from Sit to Stand
            p = (t - 2.0) / 3.0
            progress = 0.5 - 0.5 * math.cos(p * math.pi) # Smooth ease-in/ease-out curve
            
            current_hip = SIT_HIP + (STAND_HIP - SIT_HIP) * progress
            current_knee = SIT_KNEE + (STAND_KNEE - SIT_KNEE) * progress
            current_ankle = SIT_ANKLE + (STAND_ANKLE - SIT_ANKLE) * progress
            
        else:
            # After 5.0s: Lock servos in the standing position
            current_hip, current_knee, current_ankle = STAND_HIP, STAND_KNEE, STAND_ANKLE

        # Send commands to Right Leg
        data.ctrl[r_hip] = current_hip
        data.ctrl[r_knee] = current_knee
        data.ctrl[r_ankle] = current_ankle

        # Send commands to Left Leg
        data.ctrl[l_hip] = current_hip
        data.ctrl[l_knee] = current_knee
        data.ctrl[l_ankle] = current_ankle

        # Step simulation physics
        mujoco.mj_step(model, data)
        viewer.sync()

        # Enforce real-time display speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)