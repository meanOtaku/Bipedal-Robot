import pandas as pd
import numpy as np
import pygame
import sys
import os

# --- Configurations ---
CSV_FILENAME = "universal_3d_mocap_dataset.csv" # Updated to your new dataset name
WINDOW_WIDTH = 1200  
WINDOW_HEIGHT = 700  
FPS = 30  

# How many pixels represents 1 meter in the real world
METER_TO_PIXEL_SCALE = 400 

if not os.path.exists(CSV_FILENAME):
    print(f"[!] Target error: '{CSV_FILENAME}' not found. Please record data first.")
    sys.exit()

df = pd.read_csv(CSV_FILENAME)
total_frames = len(df)

if total_frames == 0:
    print("\n[!] ERROR: CSV contains no rows.")
    sys.exit()

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("3D Spatial Kinematic Telemetry Engine")
clock = pygame.time.Clock()

# --- Aesthetic Palette ---
COLOR_BACKGROUND = (20, 24, 30)
COLOR_TEXT = (200, 210, 220)
COLOR_TEXT_MUTED = (110, 120, 130)
COLOR_LEFT = (0, 230, 240)      # Cyan 
COLOR_RIGHT = (240, 0, 240)     # Magenta
COLOR_TORSO = (236, 240, 241)   # Off-White
COLOR_GROUND = (70, 80, 95)
GROUND_Y = 600  

font = pygame.font.SysFont("Consolas", 14)
font_bold = pygame.font.SysFont("Consolas", 15, bold=True)
frame_index = 0
playback_paused = False

def calculate_3d_angle(a, b, c):
    """Calculates the 3D joint angle in degrees on the fly."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

running = True
while running:
    screen.fill(COLOR_BACKGROUND)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                playback_paused = not playback_paused
            if event.key == pygame.K_q:
                running = False

    row = df.iloc[frame_index]
    
    # --- HELPER: Extract 3D Vector from CSV ---
    def get_vec3(name):
        return np.array([row[f"{name}_X"], row[f"{name}_Y"], row[f"{name}_Z"]])

    # Extract raw 3D vectors
    l_hip = get_vec3("LEFT_HIP")
    r_hip = get_vec3("RIGHT_HIP")
    l_knee = get_vec3("LEFT_KNEE")
    r_knee = get_vec3("RIGHT_KNEE")
    l_ankle = get_vec3("LEFT_ANKLE")
    r_ankle = get_vec3("RIGHT_ANKLE")
    l_heel = get_vec3("LEFT_HEEL")
    r_heel = get_vec3("RIGHT_HEEL")
    l_toe = get_vec3("LEFT_FOOT_INDEX")
    r_toe = get_vec3("RIGHT_FOOT_INDEX")
    
    l_sh = get_vec3("LEFT_SHOULDER")
    r_sh = get_vec3("RIGHT_SHOULDER")
    l_el = get_vec3("LEFT_ELBOW")
    r_el = get_vec3("RIGHT_ELBOW")
    l_wr = get_vec3("LEFT_WRIST")
    r_wr = get_vec3("RIGHT_WRIST")
    nose = get_vec3("NOSE")

    # --- 1. DYNAMIC GROUNDING ENGINE ---
    # Find the lowest Y value in the dataset (highest numerical value since Y goes down)
    lowest_y_meter = max(l_heel[1], r_heel[1], l_toe[1], r_toe[1])
    
    # Calculate the pixel offsets to center the skeleton horizontally and ground it vertically
    RENDER_OFFSET_X = 650
    RENDER_OFFSET_Y = GROUND_Y - int(lowest_y_meter * METER_TO_PIXEL_SCALE)

    # --- HELPER: Map 3D Meter coordinates to 2D Screen Pixels ---
    def to_screen(vec3):
        x = RENDER_OFFSET_X + int(vec3[0] * METER_TO_PIXEL_SCALE)
        y = RENDER_OFFSET_Y + int(vec3[1] * METER_TO_PIXEL_SCALE)
        return (x, y)

    # --- 2. RENDER SKELETON ---
    pygame.draw.line(screen, COLOR_GROUND, (350, GROUND_Y), (WINDOW_WIDTH - 20, GROUND_Y), 4)
    
    # Torso & Head
    pygame.draw.line(screen, COLOR_TORSO, to_screen(l_sh), to_screen(r_sh), 6) 
    pygame.draw.line(screen, COLOR_TORSO, to_screen(l_hip), to_screen(r_hip), 6) 
    # Spine (Mid-shoulder to Mid-hip)
    mid_sh = to_screen((l_sh + r_sh) / 2.0)
    mid_hip = to_screen((l_hip + r_hip) / 2.0)
    pygame.draw.line(screen, COLOR_TORSO, mid_sh, mid_hip, 7)
    pygame.draw.circle(screen, COLOR_TORSO, to_screen(nose), 18, 5) 
    
    # Left Arm (Cyan)
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_sh), to_screen(l_el), 5) 
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_el), to_screen(l_wr), 4) 
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_sh), 6)
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_el), 5)
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_wr), 5)
    
    # Right Arm (Magenta)
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_sh), to_screen(r_el), 5) 
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_el), to_screen(r_wr), 4) 
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_sh), 6)
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_el), 5)
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_wr), 5)
    
    # Left Leg (Cyan)
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_hip), to_screen(l_knee), 6) 
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_knee), to_screen(l_ankle), 5) 
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_ankle), to_screen(l_toe), 5)
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_ankle), to_screen(l_heel), 5)
    pygame.draw.line(screen, COLOR_LEFT, to_screen(l_heel), to_screen(l_toe), 3) # Sole of foot
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_hip), 5) 
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_knee), 6) 
    pygame.draw.circle(screen, COLOR_LEFT, to_screen(l_ankle), 6) 
    
    # Right Leg (Magenta)
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_hip), to_screen(r_knee), 6) 
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_knee), to_screen(r_ankle), 5) 
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_ankle), to_screen(r_toe), 5)
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_ankle), to_screen(r_heel), 5)
    pygame.draw.line(screen, COLOR_RIGHT, to_screen(r_heel), to_screen(r_toe), 3) # Sole of foot
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_hip), 5) 
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_knee), 6) 
    pygame.draw.circle(screen, COLOR_RIGHT, to_screen(r_ankle), 6) 

    # --- 3. DYNAMIC HUD CALCULATIONS ---
    # Calculate angles on the fly using true 3D spatial vectors
    lk_ang = calculate_3d_angle(l_hip, l_knee, l_ankle)
    rk_ang = calculate_3d_angle(r_hip, r_knee, r_ankle)
    la_ang = calculate_3d_angle(l_knee, l_ankle, l_toe)
    ra_ang = calculate_3d_angle(r_knee, r_ankle, r_toe)
    lel_ang = calculate_3d_angle(l_sh, l_el, l_wr)
    rel_ang = calculate_3d_angle(r_sh, r_el, r_wr)
    
    # Determine which foot is currently planted on the ground
    l_grounded = "YES" if l_heel[1] > r_heel[1] else "NO"
    r_grounded = "YES" if r_heel[1] > l_heel[1] else "NO"

    pygame.draw.rect(screen, (30, 36, 44), (15, 15, 310, 670))
    hud_metrics = [
        ("SYSTEM REPLAY INDEX", f"{frame_index + 1} / {total_frames}", COLOR_TEXT),
        ("TIMESTAMP (SEC)", f"{row['timestamp']:.2f}s", COLOR_TEXT),
        ("----------------------------", "", COLOR_TEXT_MUTED),
        ("SPATIAL Z-DEPTH METRICS", "", COLOR_TORSO),
        ("L_FOOT DEPTH (Z)", f"{l_ankle[2]:.2f} m", COLOR_LEFT),
        ("R_FOOT DEPTH (Z)", f"{r_ankle[2]:.2f} m", COLOR_RIGHT),
        ("L_HAND DEPTH (Z)", f"{l_wr[2]:.2f} m", COLOR_LEFT),
        ("R_HAND DEPTH (Z)", f"{r_wr[2]:.2f} m", COLOR_RIGHT),
        ("----------------------------", "", COLOR_TEXT_MUTED),
        ("UPPER BODY KINEMATICS", "", COLOR_TEXT),
        ("LEFT ELBOW ANGLE", f"{lel_ang:.1f} deg", COLOR_LEFT),
        ("RIGHT ELBOW ANGLE", f"{rel_ang:.1f} deg", COLOR_RIGHT),
        ("----------------------------", "", COLOR_TEXT_MUTED),
        ("LOWER BODY KINEMATICS", "", COLOR_TEXT),
        ("LEFT KNEE ANGLE", f"{lk_ang:.1f} deg", COLOR_LEFT),
        ("RIGHT KNEE ANGLE", f"{rk_ang:.1f} deg", COLOR_RIGHT),
        ("LEFT ANKLE ANGLE", f"{la_ang:.1f} deg", COLOR_LEFT),
        ("RIGHT ANKLE ANGLE", f"{ra_ang:.1f} deg", COLOR_RIGHT),
        ("----------------------------", "", COLOR_TEXT_MUTED),
        ("CONTACT DYNAMICS", "", COLOR_TORSO),
        ("LEFT FOOT GROUNDED", f"{l_grounded}", COLOR_LEFT),
        ("RIGHT FOOT GROUNDED", f"{r_grounded}", COLOR_RIGHT),
    ]
    
    for idx, (label, val, col) in enumerate(hud_metrics):
        if "---" in label:
            screen.blit(font.render(label, True, col), (25, 25 + (idx * 23)))
        elif val == "":
            screen.blit(font_bold.render(label, True, col), (25, 25 + (idx * 23)))
        else:
            screen.blit(font.render(label, True, col), (25, 25 + (idx * 23)))
            screen.blit(font_bold.render(val, True, col), (185, 25 + (idx * 23)))

    menu_txt = font.render("[SPACEBAR]: Pause/Play Replay   |   [Q]: Shutdown Rig", True, COLOR_TEXT_MUTED)
    screen.blit(menu_txt, (350, 665))
    pygame.display.flip()
    
    if not playback_paused:
        frame_index = (frame_index + 1) % total_frames
    clock.tick(FPS)

pygame.quit()