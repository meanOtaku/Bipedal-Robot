import cv2
import mediapipe as mp
import csv
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CSV_FILENAME = "universal_3d_mocap_dataset.csv"

# --- 1. DYNAMIC HEADER GENERATION ---
# This automatically creates headers for all 33 joints (Nose to Foot Index)
# Storing X, Y, Z (in meters) and V (Visibility confidence) for every single point
header = ["timestamp"]
for landmark in mp_pose.PoseLandmark:
    header.extend([f"{landmark.name}_X", f"{landmark.name}_Y", f"{landmark.name}_Z", f"{landmark.name}_V"])

cap = cv2.VideoCapture(0)

print("[INFO] Initializing Camera...")
print("[INFO] Recording COMPLETE 33-Point FULL-BODY dataset...")
print("[INFO] Press 'q' to stop recording and save.")

start_time = time.time()

# --- 2. FIX: OPEN FILE ONCE ---
# Opening the file using 'with' OUTSIDE the while loop prevents HDD bottlenecking.
# It writes continuously to memory and saves flawlessly when you quit.
with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("[ERROR] Camera feed lost.")
                break
            
            current_timestamp = time.time() - start_time
            
            # Flip the frame for a mirror effect, then process
            frame = cv2.flip(frame, 1)  
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # --- 3. FIX: EXTRACT TRUE 3D METRIC DATA ---
            # pose_world_landmarks gives true 3D spatial coordinates in meters, 
            # rather than flat 2D camera pixels.
            if results.pose_world_landmarks:
                row_data = [current_timestamp]
                
                # Loop through all 33 points and append their data to the row
                for landmark in results.pose_world_landmarks.landmark:
                    row_data.extend([
                        round(landmark.x, 6), 
                        round(landmark.y, 6), 
                        round(landmark.z, 6), 
                        round(landmark.visibility, 6)
                    ])
                
                # Write the massive 133-column row to the CSV instantly
                writer.writerow(row_data)

                # Draw the skeletal mesh on the video feed
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, # We draw the 2D ones so it matches the screen
                    mp_pose.POSE_CONNECTIONS
                )               

            # Display recording status on the screen
            cv2.putText(image, f"RECORDING 3D MOCAP - Time: {current_timestamp:.1f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Kinematic Data Collection', image)
            
            # Press 'q' to break the loop safely
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print("[INFO] Recording stopped by user.")
                break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"[SUCCESS] Dataset saved to {CSV_FILENAME}")