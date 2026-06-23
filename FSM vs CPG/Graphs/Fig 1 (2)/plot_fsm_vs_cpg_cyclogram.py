import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fsm_vs_cpg_single_cyclogram():
    plt.style.use('seaborn-v0_8-deep')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [
        ('Finite-State Keyframe (FSM)', 'telemetry_forward.csv', '-', 'crimson'),
        ('Continuous Phase Manifold (CPG)', 'telemetry_cpg_1.0.csv', '-', 'midnightblue')
    ]
    
    data_found = False
    
    for label, filename, linestyle, color in datasets:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Filter for steady state walking
            df = df[df['time'] > 2.0]
            
            if 'left_hip_pitch_pos_rad' in df.columns and 'left_knee_pitch_pos_rad' in df.columns:
                # Hip Angle (Flexion Positive)
                hip = -np.degrees(df['left_hip_pitch_pos_rad']) 
                # Knee Angle
                knee = np.abs(np.degrees(df['left_knee_pitch_pos_rad']))
                
                ax.plot(hip, knee, label=label, 
                        color=color, linestyle=linestyle, linewidth=2.5, alpha=0.9)
                data_found = True
        else:
            print(f"Warning: {filename} not found.")

    if not data_found:
        print("No telemetry data found to plot!")
        return

    # Styling to match the strict PI rules
    ax.set_title('Hip vs Knee Angle Cyclogram', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Hip Angle (deg) - Flexion Positive', fontsize=11, fontweight='bold')
    ax.set_ylabel('Knee Angle (deg)', fontsize=11, fontweight='bold')
    
    # PI's strict grid formatting
    ax.grid(True, linestyle='--', linewidth=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    
    plt.tight_layout()
    
    output_pdf = 'fsm_vs_cpg_single_cyclogram.pdf'
    output_svg = 'fsm_vs_cpg_single_cyclogram.svg'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', format='svg')
    print(f"Vector graphs saved to {output_pdf} and {output_svg}")

if __name__ == "__main__":
    plot_fsm_vs_cpg_single_cyclogram()
