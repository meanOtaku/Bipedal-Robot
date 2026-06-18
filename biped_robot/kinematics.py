import mujoco
import numpy as np


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


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


def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def get_imu_data(model, data):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    adr = model.sensor_adr[sensor_id]
    quat = data.sensordata[adr:adr + 4]
    return quat_to_euler(quat)


def get_foot_surface_point(model, data, geom_name):
    gid = geom_id(model, geom_name)
    center = data.geom_xpos[gid]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    half_height = model.geom_size[gid, 2]
    return center + rotation @ np.array([0.0, 0.0, -half_height])


def world_to_root_frame(data, point):
    root_pos = data.qpos[0:3]
    root_quat = data.qpos[3:7]
    root_rotation = quat_to_rotation_matrix(root_quat)
    return root_rotation.T @ (point - root_pos)

