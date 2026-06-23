import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import os

def extract_stance_power(filename):
    df = pd.read_csv(filename)
    df = df[df['time'] > 2.0] # steady state
    
    # RPM to rad/s
    rad_s = df['left_ankle_pitch_vel_rpm'] * (np.pi / 30.0)
    power = df['left_ankle_pitch_torque_nm'] * rad_s
    
    # Find the most negative power peak to center our window on the "trailing stance" push-off
    min_idx = power.argmin()
    t_center = df['time'].iloc[min_idx]
    
    # Extract 0.5 second window
    mask = (df['time'] >= t_center - 0.15) & (df['time'] <= t_center + 0.35)
    df_win = df[mask].copy()
    
    t_norm = df_win['time'].values - df_win['time'].values[0]
    p_win = power.loc[mask].values
    
    # Calculate integral of parasitic work (negative power only)
    negative_power = np.minimum(0, p_win)
    parasitic_work = scipy.integrate.simpson(negative_power, x=t_norm)
    
    return t_norm, p_win, parasitic_work

def plot_parasitic_work():
    plt.style.use('seaborn-v0_8-deep')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    fsm_color = 'crimson'
    cpg_color = 'midnightblue'
    
    grid_lw = 0.3
    grid_ls = '--'
    
    fsm_file = 'telemetry_forward.csv'
    cpg_file = 'telemetry_cpg_1.0.csv'
    
    try:
        t_fsm, p_fsm, w_fsm = extract_stance_power(fsm_file)
        axes[0].plot(t_fsm, p_fsm, color=fsm_color, linewidth=2.5, label='Live Power')
        axes[0].axhline(0, color='black', linewidth=1)
        
        # Shade negative area
        axes[0].fill_between(t_fsm, p_fsm, 0, where=(p_fsm < 0), 
                             color='red', alpha=0.3, label='Parasitic Negative Work')
                             
        axes[0].set_title(f'Finite-State Keyframe (FSM)\n$W_{{parasitic}}$ = {abs(w_fsm):.2f} Joules', fontsize=12)
        axes[0].set_xlabel('Time within stance phase (s)', fontsize=11)
        axes[0].set_ylabel('Mechanical Power (Watts)', fontsize=11)
        axes[0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    except Exception as e:
        print(f"Error on FSM: {e}")

    try:
        t_cpg, p_cpg, w_cpg = extract_stance_power(cpg_file)
        axes[1].plot(t_cpg, p_cpg, color=cpg_color, linewidth=2.5, label='Live Power')
        axes[1].axhline(0, color='black', linewidth=1)
        
        # Shade negative area
        axes[1].fill_between(t_cpg, p_cpg, 0, where=(p_cpg < 0), 
                             color='red', alpha=0.3, label='Parasitic Negative Work')
                             
        axes[1].set_title(f'Continuous Phase Manifold (CPG)\n$W_{{parasitic}}$ = {abs(w_cpg):.2f} Joules', fontsize=12)
        axes[1].set_xlabel('Time within stance phase (s)', fontsize=11)
        axes[1].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    except Exception as e:
        print(f"Error on CPG: {e}")

    # Apply strict grid formatting to both axes
    for ax in axes:
        ax.grid(True, linestyle=grid_ls, linewidth=grid_lw, color='gray')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.suptitle('Figure 3: Parasitic Work Deconstruction', fontsize=16, fontweight='bold', y=0.98)
    
    # 15% bottom margin to protect the X-axis from the caption box
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Weaponized Caption
    caption = "Fig. 3. Instantaneous mechanical power dissipation at the trailing ankle joint during late stance.\n" \
              "Shaded regions denote parasitic negative work ($W_{parasitic}$) caused by the rigid spatial tracking of static dictionary keyframes."
              
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.5'))
             
    output_pdf = 'parasitic_work.pdf'
    output_svg = 'parasitic_work.svg'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', format='svg')
    print(f"Vector graphs saved to {output_pdf} and {output_svg}")

if __name__ == "__main__":
    plot_parasitic_work()
