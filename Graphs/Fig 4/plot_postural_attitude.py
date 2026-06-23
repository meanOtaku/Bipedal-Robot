import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_postural_attitude():
    # PI's Styling Rules
    plt.style.use('seaborn-v0_8-deep')  # Use seaborn deep palette as a base
    
    # Specific colors demanded by PI
    fsm_color = 'crimson'      # Symbolizing danger, rigidity, high energy, failure
    cpg_color = 'midnightblue' # Deep Navy Blue / Slate, symbolizing optimized mathematics
    
    # Grid settings demanded by PI
    grid_lw = 0.3
    grid_ls = '--'
    
    datasets = [
        ('Finite-State Keyframe (FSM)', 'telemetry_forward.csv', '-', fsm_color),
        ('Continuous Phase Manifold (CPG)', 'telemetry_cpg_1.0.csv', '-', cpg_color)
    ]
    
    # 10-second walking strip
    WINDOW_START = 3.0
    WINDOW_DURATION = 10.0
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    data_found = False
    
    for label, filename, linestyle, color in datasets:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            
            mask = (df['time'] >= WINDOW_START) & (df['time'] <= WINDOW_START + WINDOW_DURATION)
            df_win = df[mask].copy()
            
            if not df_win.empty:
                data_found = True
                t = df_win['time'] - df_win['time'].iloc[0]
                
                torso_z = df_win['torso_z']
                # The prompt asks for imu_pitch, our header is torso_pitch (which is the same)
                imu_pitch = df_win['torso_pitch']
                
                # Top Subplot: torso_z
                axs[0].plot(t, torso_z, label=label, color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)
                
                # Bottom Subplot: imu_pitch
                axs[1].plot(t, imu_pitch, label=label, color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)
        else:
            print(f"Warning: {filename} not found.")

    if not data_found:
        print("No telemetry data found!")
        return

    # Titles and Labels
    fig.suptitle('Figure 4: The Postural Attitude Platform', fontsize=15, fontweight='bold', y=0.95)
    
    axs[0].set_ylabel('CoM Z-Position (m)', fontsize=11, fontweight='bold')
    axs[0].set_title('Vertical Center of Mass (CoM) Displacement', fontsize=12)
    axs[0].legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', fontsize=10)
    
    axs[1].set_ylabel('Vestibular Attitude / Pitch (rad)', fontsize=11, fontweight='bold')
    axs[1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    axs[1].set_title('IMU Pitch Oscillation', fontsize=12)
    
    # PI's strict grid requirements
    for ax in axs:
        ax.grid(True, linestyle=grid_ls, linewidth=grid_lw, color='gray')
        
        # Make spines a bit cleaner for scientific publication
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # Leave 15% empty space at the bottom for the caption box to sit cleanly
    plt.tight_layout(rect=[0, 0.15, 1, 0.93])
    
    # Weaponized Caption
    caption = "Fig. 4. Center of Mass (CoM) platform stabilization.\n" \
              "Continuous phase progression eliminates the mid-stance vertical vaulting characteristic of rigid wooden-peg bipeds,\n" \
              "stabilizing the vestibular attitude."
              
    # Place the caption inside the empty bottom margin we just created
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey', boxstyle='round,pad=0.5'))
             
    # PI's strict format requirement: PDF and SVG ONLY
    output_pdf = 'postural_attitude.pdf'
    output_svg = 'postural_attitude.svg'
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', format='svg')
    
    print(f"Graphs saved as true vectors to {output_pdf} and {output_svg}")

if __name__ == "__main__":
    plot_postural_attitude()
