import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mujoco

def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def compute_manifold(filename):
    model = mujoco.MjModel.from_xml_path("biped.xml")
    data = mujoco.MjData(model)
    
    df = pd.read_csv(filename)
    df = df[df['time'] > 2.0]
    
    joint_names = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
    ]
    
    x_local_list = []
    z_list = []
    
    rf_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    
    for _, row in df.iterrows():
        # Set torso translation
        data.qpos[0] = row['torso_x']
        data.qpos[1] = row['torso_y']
        data.qpos[2] = row['torso_z']
        
        # Set torso rotation
        quat = euler_to_quat(row['torso_roll'], row['torso_pitch'], row['torso_yaw'])
        data.qpos[3:7] = quat
        
        # Set joints
        for jn in joint_names:
            col = f"{jn}_pos_rad"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            qpos_addr = model.jnt_qposadr[jid]
            data.qpos[qpos_addr] = row[col]
            
        mujoco.mj_kinematics(model, data)
        
        right_foot_x = data.xpos[rf_body_id, 0]
        right_foot_z = data.xpos[rf_body_id, 2]
        
        x_local = right_foot_x - row['torso_x']
        # For Z, if we want purely local Z: right_foot_z - torso_z. 
        # But commonly the "kinematic manifold" plots local X vs local Z.
        z_local = right_foot_z - row['torso_z']
        
        x_local_list.append(x_local)
        z_list.append(z_local)
        
    return x_local_list, z_list

def plot_manifold():
    plt.style.use('seaborn-v0_8-deep')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    fsm_color = 'crimson'
    cpg_color = 'midnightblue'
    
    grid_lw = 0.3
    grid_ls = '--'
    
    fsm_file = 'telemetry_forward.csv'
    cpg_file = 'telemetry_cpg_1.0.csv'
    
    try:
        fsm_x, fsm_z = compute_manifold(fsm_file)
        axes[0].plot(fsm_x, fsm_z, color=fsm_color, linewidth=2.5)
        axes[0].set_title('Finite-State Keyframe (FSM)', fontsize=13)
        axes[0].set_xlabel('$x_{local}$ (m)', fontsize=12)
        axes[0].set_ylabel('$z_{local}$ (m)', fontsize=12)
    except Exception as e:
        print(f"Error on FSM: {e}")
        
    try:
        cpg_x, cpg_z = compute_manifold(cpg_file)
        axes[1].plot(cpg_x, cpg_z, color=cpg_color, linewidth=2.5)
        axes[1].set_title('Continuous Phase Manifold (CPG)', fontsize=13)
        axes[1].set_xlabel('$x_{local}$ (m)', fontsize=12)
    except Exception as e:
        print(f"Error on CPG: {e}")
        
    # Apply strict grid formatting to both axes
    for ax in axes:
        ax.grid(True, linestyle=grid_ls, linewidth=grid_lw, color='gray')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.suptitle('Figure 1: The Kinematic Manifold', fontsize=16, fontweight='bold', y=0.98)
    
    # 15% bottom margin to protect the X-axis from the caption box
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Weaponized Caption
    caption = "Fig. 1. Cartesian swing-foot phase portraits in the sagittal plane (x-z).\n" \
              "The discrete Finite-State Keyframe approach (left) yields non-differentiable spatial vertices,\n" \
              "whereas the continuous CPG phase manifold (right) generates a C^2-continuous, smooth limit cycle."
              
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.5'))
             
    output_pdf = 'kinematic_manifold.pdf'
    output_svg = 'kinematic_manifold.svg'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', format='svg')
    print(f"Vector graphs saved to {output_pdf} and {output_svg}")

if __name__ == "__main__":
    plot_manifold()
