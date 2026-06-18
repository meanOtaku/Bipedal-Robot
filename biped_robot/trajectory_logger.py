import collections
import csv

import mujoco
import numpy as np

from .config import PROJECT_ROOT
from .kinematics import body_id, geom_id, get_foot_surface_point, get_imu_data, world_to_root_frame


class TrajectoryLogger:
    def __init__(self, model, data, joint_names, body_names, foot_geom_names):
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.body_names = body_names
        self.foot_geom_names = foot_geom_names
        self.telemetry_data = collections.defaultdict(list)
        self.rows = []
        self.headers = self._build_headers()
        self.path = PROJECT_ROOT / "robot_trajectory.csv"
        self.file = self.path.open("w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        print(f"Recording live trajectory to {self.path}")

    def _build_headers(self):
        headers = [
            "time", "state", "mode", "command", "step_count",
            "root_x", "root_y", "root_z",
            "torso_x", "torso_y", "torso_z",
            "com_x", "com_y", "com_z",
            "imu_roll", "imu_pitch", "imu_yaw",
        ]
        for body_name in self.body_names:
            headers.extend([f"{body_name}_x", f"{body_name}_y", f"{body_name}_z"])
        for geom_name in self.foot_geom_names:
            headers.extend([
                f"{geom_name}_center_x", f"{geom_name}_center_y", f"{geom_name}_center_z",
                f"{geom_name}_surface_x", f"{geom_name}_surface_y", f"{geom_name}_surface_z",
                f"{geom_name}_ground_x", f"{geom_name}_ground_y", f"{geom_name}_ground_z",
                f"{geom_name}_root_rel_x", f"{geom_name}_root_rel_y", f"{geom_name}_root_rel_z",
                f"{geom_name}_ground_root_rel_x", f"{geom_name}_ground_root_rel_y", f"{geom_name}_ground_root_rel_z",
            ])
        for joint_name in self.joint_names:
            headers.extend([
                f"{joint_name}_target_rad",
                f"{joint_name}_actual_rad",
                f"{joint_name}_vel_rpm",
                f"{joint_name}_torque_nm",
            ])
        return headers

    def record(self, elapsed, state, walk_mode, command, step_count, target_joint_positions, current_action):
        torso_id = body_id(self.model, "torso")
        com = self.data.subtree_com[torso_id]
        roll, pitch, yaw = get_imu_data(self.model, self.data)
        row = [
            elapsed, state, walk_mode, command, step_count,
            self.data.qpos[0], self.data.qpos[1], self.data.qpos[2],
            self.data.xpos[torso_id, 0], self.data.xpos[torso_id, 1], self.data.xpos[torso_id, 2],
            com[0], com[1], com[2],
            roll, pitch, yaw,
        ]

        for body_name in self.body_names:
            row.extend(self.data.xpos[body_id(self.model, body_name)].tolist())

        for geom_name in self.foot_geom_names:
            gid = geom_id(self.model, geom_name)
            surface_point = get_foot_surface_point(self.model, self.data, geom_name)
            ground_point = np.array([surface_point[0], surface_point[1], 0.0])
            row.extend(self.data.geom_xpos[gid].tolist())
            row.extend(surface_point.tolist())
            row.extend(ground_point.tolist())
            row.extend(world_to_root_frame(self.data, surface_point).tolist())
            row.extend(world_to_root_frame(self.data, ground_point).tolist())

        for actuator_id, joint_name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            pos = self.data.qpos[self.model.jnt_qposadr[jid]]
            vel = self.data.qvel[self.model.jnt_dofadr[jid]]
            rpm = vel * 60.0 / (2 * np.pi)
            gear = self.model.actuator_gear[actuator_id, 0]
            actual_torque = self.data.ctrl[actuator_id] * gear
            row.extend([target_joint_positions[joint_name], pos, rpm, actual_torque])

        self.telemetry_data[current_action].append(row)
        self.rows.append(row)
        self.writer.writerow(row)
        return row

    def flush(self):
        self.file.flush()

    def close(self):
        self.flush()
        self.file.close()
        print("Simulation finished. Writing trajectory files...")
        if self.rows:
            print(f"Saved {len(self.rows)} rows to {self.path}.")
        for action, data_rows in self.telemetry_data.items():
            if not data_rows:
                continue
            path = PROJECT_ROOT / f"telemetry_{action}.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(data_rows)
            print(f"Saved {len(data_rows)} rows of telemetry data to {path}.")

