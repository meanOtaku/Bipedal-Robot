import cv2
import mediapipe as mp
import math
import xml.etree.ElementTree as ET
import os
import pygame
import numpy as np

# --- Configuration ---
XML_FILENAME = "biped.xml"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_dist(p1, p2):
    """Calculate Euclidean distance between two 3D landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def update_xml(thigh_len, shin_len):
    """Update biped.xml with the new limb lengths."""
    if not os.path.exists(XML_FILENAME):
        print(f"\n[ERROR] {XML_FILENAME} not found.")
        return

    tree = ET.parse(XML_FILENAME)
    root = tree.getroot()

    def find_by_name(element, tag, name):
        for child in element.iter(tag):
            if child.get('name') == name:
                return child
        return None

    # Update Left Leg
    left_thigh_geom = find_by_name(root, 'geom', 'left_thigh')
    if left_thigh_geom is not None: left_thigh_geom.set('fromto', f"0 0 0 0 0 -{thigh_len:.3f}")
    left_shin_link = find_by_name(root, 'body', 'left_shin_link')
    if left_shin_link is not None: left_shin_link.set('pos', f"0 0 -{thigh_len:.3f}")
    left_shin_geom = find_by_name(root, 'geom', 'left_shin')
    if left_shin_geom is not None: left_shin_geom.set('fromto', f"0 0 0 0 0 -{shin_len:.3f}")
    left_ankle_link = find_by_name(root, 'body', 'left_ankle_pitch_link')
    if left_ankle_link is not None: left_ankle_link.set('pos', f"0 0 -{shin_len:.3f}")

    # Update Right Leg
    right_thigh_geom = find_by_name(root, 'geom', 'right_thigh')
    if right_thigh_geom is not None: right_thigh_geom.set('fromto', f"0 0 0 0 0 -{thigh_len:.3f}")
    right_shin_link = find_by_name(root, 'body', 'right_shin_link')
    if right_shin_link is not None: right_shin_link.set('pos', f"0 0 -{thigh_len:.3f}")
    right_shin_geom = find_by_name(root, 'geom', 'right_shin')
    if right_shin_geom is not None: right_shin_geom.set('fromto', f"0 0 0 0 0 -{shin_len:.3f}")
    right_ankle_link = find_by_name(root, 'body', 'right_ankle_pitch_link')
    if right_ankle_link is not None: right_ankle_link.set('pos', f"0 0 -{shin_len:.3f}")

    tree.write(XML_FILENAME, encoding='utf-8', xml_declaration=True)
    print(f"\n[SUCCESS] Updated {XML_FILENAME} -> Thigh: {thigh_len:.3f}m, Shin: {shin_len:.3f}m")

def main():
    print("[INFO] Initializing PyGame GUI and Camera...")
    
    # Init Pygame
    pygame.init()
    cap = cv2.VideoCapture(0)
    
    # Get webcam resolution
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from webcam.")
        return
        
    h, w, _ = frame.shape
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Limb Length Estimator (Auto-Capture)")
    
    font = pygame.font.SysFont("Arial", 24, bold=True)
    large_font = pygame.font.SysFont("Arial", 48, bold=True)

    avg_thigh = 0.40
    avg_shin = 0.40
    alpha = 0.1 # Smoothing factor

    # Auto-capture logic
    history_thigh = []
    history_shin = []
    REQUIRED_FRAMES = 40       # Must be visible and stable for ~1.5 seconds
    STABILITY_THRESHOLD = 0.05 # Fluctuation must be less than 5 cm
    auto_saved = False
    save_timer = 0

    clock = pygame.time.Clock()
    running = True

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while running and cap.isOpened():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with mediapipe
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                
                # Get points
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # Calculate distances (in meters)
                l_thigh = get_dist(l_hip, l_knee)
                l_shin = get_dist(l_knee, l_ankle)
                r_thigh = get_dist(r_hip, r_knee)
                r_shin = get_dist(r_knee, r_ankle)

                curr_thigh = (l_thigh + r_thigh) / 2.0
                curr_shin = (l_shin + r_shin) / 2.0

                # Smooth the visual values
                avg_thigh = (alpha * curr_thigh) + ((1.0 - alpha) * avg_thigh)
                avg_shin = (alpha * curr_shin) + ((1.0 - alpha) * avg_shin)

                if not auto_saved:
                    history_thigh.append(curr_thigh)
                    history_shin.append(curr_shin)
                    
                    if len(history_thigh) > REQUIRED_FRAMES:
                        history_thigh.pop(0)
                        history_shin.pop(0)
                        
                    if len(history_thigh) == REQUIRED_FRAMES:
                        thigh_range = max(history_thigh) - min(history_thigh)
                        shin_range = max(history_shin) - min(history_shin)
                        
                        # Check if person is standing still
                        if thigh_range < STABILITY_THRESHOLD and shin_range < STABILITY_THRESHOLD:
                            final_thigh = sum(history_thigh) / REQUIRED_FRAMES
                            final_shin = sum(history_shin) / REQUIRED_FRAMES
                            update_xml(final_thigh, final_shin)
                            auto_saved = True
                            save_timer = pygame.time.get_ticks()

                # Draw stick transition (skeleton) ON THE FRAME
                mp_drawing.draw_landmarks(
                    image_rgb, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
            else:
                # If tracking is lost, clear history
                if not auto_saved:
                    history_thigh.clear()
                    history_shin.clear()

            # Convert RGB image to PyGame surface
            frame_surface = np.transpose(image_rgb, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_surface)
            screen.blit(surf, (0, 0))

            # Overlay UI
            if auto_saved:
                # Show success message
                bg_rect = pygame.Rect(w//2 - 200, h//2 - 50, 400, 100)
                pygame.draw.rect(screen, (0, 200, 0), bg_rect, border_radius=10)
                text_success = large_font.render("CAPTURED & SAVED!", True, (255, 255, 255))
                screen.blit(text_success, (w//2 - text_success.get_width()//2, h//2 - text_success.get_height()//2))
                
                # Auto quit after 3 seconds
                if pygame.time.get_ticks() - save_timer > 3000:
                    running = False
            else:
                # Show live stats and progress bar
                text1 = font.render(f"Thigh: {avg_thigh:.3f}m", True, (0, 255, 0))
                text2 = font.render(f"Shin:  {avg_shin:.3f}m", True, (0, 255, 0))
                screen.blit(text1, (20, 20))
                screen.blit(text2, (20, 50))
                
                # Auto-capture progress bar
                if len(history_thigh) > 0:
                    status_text = font.render("Stand Still to Auto-Capture...", True, (255, 255, 0))
                    screen.blit(status_text, (20, h - 80))
                    
                    bar_width = 300
                    fill_width = int((len(history_thigh) / REQUIRED_FRAMES) * bar_width)
                    pygame.draw.rect(screen, (100, 100, 100), (20, h - 40, bar_width, 20))
                    pygame.draw.rect(screen, (0, 255, 0), (20, h - 40, fill_width, 20))
                else:
                    status_text = font.render("Waiting for person in frame...", True, (255, 100, 100))
                    screen.blit(status_text, (20, h - 80))

            pygame.display.flip()
            clock.tick(30)

    cap.release()
    pygame.quit()
    print("[INFO] Quitting.")

if __name__ == "__main__":
    main()