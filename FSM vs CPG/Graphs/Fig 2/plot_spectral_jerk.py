import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import os

def compute_psd(filename):
    df = pd.read_csv(filename)
    # Filter steady state walking
    df = df[df['time'] > 2.0]
    
    time = df['time'].values
    torque = df['left_knee_pitch_torque_nm'].values
    
    # Calculate Actuator Jerk (discrete time-derivative of torque)
    dt = np.diff(time)
    jerk = np.diff(torque) / dt
    
    N = len(jerk)
    T = np.mean(dt)
    
    # Compute Power Spectral Density (PSD) using rfft as requested
    yf = scipy.fft.rfft(jerk)
    xf = scipy.fft.rfftfreq(N, T)
    
    # PSD = (1/N) * |FFT|^2
    psd = (1.0 / N) * np.abs(yf)**2
    
    return xf, psd

def plot_spectral_jerk():
    plt.style.use('seaborn-v0_8-deep')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fsm_color = 'crimson'
    cpg_color = 'midnightblue'
    
    fsm_file = 'telemetry_forward.csv'
    cpg_file = 'telemetry_cpg_1.0.csv'
    
    try:
        xf_fsm, psd_fsm = compute_psd(fsm_file)
        ax.plot(xf_fsm, psd_fsm, color=fsm_color, label='Finite-State Keyframe (FSM)', linewidth=2.0, alpha=0.9)
    except Exception as e:
        print(f"Error processing FSM data: {e}")
        
    try:
        xf_cpg, psd_cpg = compute_psd(cpg_file)
        ax.plot(xf_cpg, psd_cpg, color=cpg_color, label='Continuous Phase Manifold (CPG)', linewidth=2.0, alpha=0.9)
    except Exception as e:
        print(f"Error processing CPG data: {e}")

    ax.set_xlim(0, 50)
    ax.set_yscale('log')
    
    ax.set_title('Figure 2: Spectral Jerk Attenuation', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (Log-Scale)', fontsize=12)
    
    # PI's strict grid formatting
    ax.grid(True, linestyle='--', linewidth=0.3, color='gray', which='both')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        
    ax.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    
    # 15% bottom margin to protect the X-axis from the caption box
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # Weaponized Caption
    caption = "Fig. 2. Power Spectral Density (PSD) of the knee actuator jerk ($\\dot{\\tau}$).\n" \
              "Step-function transitions in the FSM inject severe high-frequency structural harmonics into the chassis,\n" \
              "which are successfully attenuated by the CPG manifold."
              
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.5'))
             
    output_pdf = 'spectral_jerk.pdf'
    output_svg = 'spectral_jerk.svg'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', format='svg')
    print(f"Vector graphs saved to {output_pdf} and {output_svg}")

if __name__ == "__main__":
    plot_spectral_jerk()
