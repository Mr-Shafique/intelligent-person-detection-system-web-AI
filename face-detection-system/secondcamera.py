import cv2
import os
from ultralytics import YOLO
import time
import numpy as np
import math

# Load YOLOv8 model
try:
    model = YOLO('yolov8n-face.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolov8n-face.pt' is in the correct location.")
    exit()

output_dir = 'capturedfaces'
os.makedirs(output_dir, exist_ok=True)

webcam = cv2.VideoCapture(0)
ip_camera_url = "http://10.102.138.81:8080/video" # Replace with your IP camera URL
ip_camera = cv2.VideoCapture(ip_camera_url)

webcam_available = webcam.isOpened()
if not webcam_available:
    print("Warning: Could not open the webcam.")

ip_camera_available = ip_camera.isOpened()
if not ip_camera_available:
    print("Warning: Could not connect to the IP camera.")

if not webcam_available and not ip_camera_available:
    print("Error: Neither webcam nor IP camera is available. Exiting.")
    if webcam is not None and webcam.isOpened(): webcam.release()
    if ip_camera is not None and ip_camera.isOpened(): ip_camera.release()
    cv2.destroyAllWindows()
    exit()

print("Press 'q' to quit.")

webcam_frame_count = 0
ipcam_frame_count = 0
PROCESS_EVERY_N_FRAMES = 1 # Process every Nth frame for detection
initial_webcam_intended = webcam_available
initial_ip_camera_intended = ip_camera_available
FACE_CLASS_ID = 0

# --- Person Counting Variables (Two-Line Method) ---
# For IN: cross Line A (upper) then Line B (lower). Y increases downwards.
# For OUT: cross Line B (lower) then Line A (upper).
# Ensure LINE_A_Y < LINE_B_Y for this logic to work as intended for IN=down, OUT=up
LINE_A_Y = 200  # Upper line: Persons moving IN cross this first (downwards)
LINE_B_Y = 300  # Lower line: Persons moving IN cross this second (downwards)

# Sanity check for line positions (optional, but good for debugging)
if LINE_A_Y >= LINE_B_Y:
    print("Warning: LINE_A_Y should be less than LINE_B_Y for the current IN/OUT logic (IN=downwards).")
    print("Please adjust line Y coordinates.")
    # You might want to exit or use default safe values if this condition is met.

PERSONS_IN_COUNT = 0
PERSONS_OUT_COUNT = 0
tracked_faces = {} # {face_id: {'centroid': (cx, cy), 'last_seen_frame': frame_count, 'crossed_A_pending_B': False, 'crossed_B_pending_A': False}}
next_face_id = 0
MAX_FRAMES_UNSEEN = 10
CENTROID_MATCH_THRESHOLD = 75 # Pixels
# --- End Person Counting Variables ---

while True:
    webcam_display_frame = None
    ipcam_display_frame = None

    # Process webcam if it's currently marked available
    if webcam_available:
        ret_webcam, frame_webcam_original = webcam.read() # Keep original for saving
        if ret_webcam:
            webcam_frame_count += 1
            resized_webcam_for_detection = cv2.resize(frame_webcam_original, (640, 480))
            webcam_display_frame = resized_webcam_for_detection.copy()
            
            current_detected_centroids_with_boxes = []
            found_face_in_webcam_for_saving = False # Flag for saving the frame

            if webcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_webcam = model(resized_webcam_for_detection, conf=0.5, verbose=False)
                
                if results_webcam and results_webcam[0].boxes.data.numel() > 0:
                    for box_data in results_webcam[0].boxes:
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                        conf = box_data.conf[0].item()
                        cls = int(box_data.cls[0].item())
                        if cls == FACE_CLASS_ID:
                            found_face_in_webcam_for_saving = True # Set flag if any face is detected
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            current_detected_centroids_with_boxes.append(((cx, cy), (x1, y1, x2, y2)))
                
                # Save frame if faces were detected in this processing cycle
                if found_face_in_webcam_for_saving:
                    timestamp = int(time.time() * 1000)
                    frame_filename = os.path.join(output_dir, f"webcam_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, frame_webcam_original) # Save original high-res frame
                    print(f"Saved webcam frame: {frame_filename}")


                # --- Two-Line Person Counting Logic for Webcam ---
                temp_current_detections = list(current_detected_centroids_with_boxes)

                for face_id, track_data in list(tracked_faces.items()):
                    best_match_idx = -1
                    min_dist = float('inf')
                    for i, (centroid, box) in enumerate(temp_current_detections):
                        dist = math.dist(track_data['centroid'], centroid)
                        if dist < CENTROID_MATCH_THRESHOLD and dist < min_dist:
                            min_dist = dist
                            best_match_idx = i
                    
                    if best_match_idx != -1: 
                        matched_centroid, matched_box = temp_current_detections.pop(best_match_idx)
                        prev_cy = track_data['centroid'][1]
                        curr_cy = matched_centroid[1]

                        tracked_faces[face_id]['centroid'] = matched_centroid
                        tracked_faces[face_id]['last_seen_frame'] = webcam_frame_count

                        # Check for crossing Line A towards B (potential IN start)
                        # Moving DOWN across Line A
                        if not tracked_faces[face_id]['crossed_A_pending_B'] and \
                           prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y:
                            tracked_faces[face_id]['crossed_A_pending_B'] = True
                            tracked_faces[face_id]['crossed_B_pending_A'] = False # Reset other direction
                            # print(f"ID {face_id} crossed Line A downwards (pending B for IN).")

                        # Check for crossing Line B towards A (potential OUT start)
                        # Moving UP across Line B
                        elif not tracked_faces[face_id]['crossed_B_pending_A'] and \
                             prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y:
                            tracked_faces[face_id]['crossed_B_pending_A'] = True
                            tracked_faces[face_id]['crossed_A_pending_B'] = False # Reset other direction
                            # print(f"ID {face_id} crossed Line B upwards (pending A for OUT).")

                        # Check for IN completion (crossed A, now crossing B downwards)
                        if tracked_faces[face_id]['crossed_A_pending_B'] and \
                           prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y:
                            PERSONS_IN_COUNT += 1
                            print(f"ID {face_id} COUNTED IN. Total IN: {PERSONS_IN_COUNT}")
                            tracked_faces[face_id]['crossed_A_pending_B'] = False # Reset for next full sequence
                            tracked_faces[face_id]['crossed_B_pending_A'] = False # Crucial reset

                        # Check for OUT completion (crossed B, now crossing A upwards)
                        elif tracked_faces[face_id]['crossed_B_pending_A'] and \
                             prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y:
                            PERSONS_OUT_COUNT += 1
                            print(f"ID {face_id} COUNTED OUT. Total OUT: {PERSONS_OUT_COUNT}")
                            tracked_faces[face_id]['crossed_B_pending_A'] = False # Reset for next full sequence
                            tracked_faces[face_id]['crossed_A_pending_B'] = False # Crucial reset
                        
                        # If person moves back over a line before completing sequence, reset pending state
                        # Moving back UP over A after pending B (was going IN, turned back)
                        if tracked_faces[face_id]['crossed_A_pending_B'] and curr_cy < LINE_A_Y:
                            tracked_faces[face_id]['crossed_A_pending_B'] = False
                            # print(f"ID {face_id} moved back UP over A, reset A_pending_B.")
                        # Moving back DOWN over B after pending A (was going OUT, turned back)
                        if tracked_faces[face_id]['crossed_B_pending_A'] and curr_cy >= LINE_B_Y:
                            tracked_faces[face_id]['crossed_B_pending_A'] = False
                            # print(f"ID {face_id} moved back DOWN over B, reset B_pending_A.")

                        (x1_trk,y1_trk,x2_trk,y2_trk) = matched_box
                        cv2.rectangle(webcam_display_frame, (x1_trk, y1_trk), (x2_trk, y2_trk), (255, 0, 0), 2)
                        cv2.putText(webcam_display_frame, f"ID:{face_id}", (x1_trk, y1_trk - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else: 
                        if (webcam_frame_count - track_data['last_seen_frame']) > MAX_FRAMES_UNSEEN:
                            # print(f"Removing lost track ID {face_id}")
                            del tracked_faces[face_id]
                
                for centroid, box in temp_current_detections:
                    new_id = next_face_id
                    next_face_id += 1
                    tracked_faces[new_id] = {
                        'centroid': centroid,
                        'last_seen_frame': webcam_frame_count,
                        'crossed_A_pending_B': False,
                        'crossed_B_pending_A': False
                    }
                    # print(f"New track ID {new_id} at {centroid}")
                    (x1_new,y1_new,x2_new,y2_new) = box
                    cv2.rectangle(webcam_display_frame, (x1_new, y1_new), (x2_new, y2_new), (0, 0, 255), 2)
                    cv2.putText(webcam_display_frame, f"ID:{new_id}", (x1_new, y1_new - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw the counting lines and counts on every frame for display
            cv2.line(webcam_display_frame, (0, LINE_A_Y), (640, LINE_A_Y), (0, 255, 0), 2)
            cv2.putText(webcam_display_frame, "A", (5, LINE_A_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.line(webcam_display_frame, (0, LINE_B_Y), (640, LINE_B_Y), (0, 0, 255), 2)
            cv2.putText(webcam_display_frame, "B", (5, LINE_B_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(webcam_display_frame, f"IN: {PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(webcam_display_frame, f"OUT: {PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # --- End Person Counting Logic ---
        else:
            print("Warning: Webcam frame read failed. Marking as unavailable.")
            if webcam is not None: webcam.release()
            webcam_available = False
            webcam_display_frame = None

    # Process IP camera if it's currently marked available
    if ip_camera_available:
        ret_ip, frame_ip_original = ip_camera.read() # Keep original for saving
        if ret_ip:
            ipcam_frame_count += 1
            resized_ip_for_detection = cv2.resize(frame_ip_original, (640, 480))
            ipcam_display_frame = resized_ip_for_detection.copy()
            
            found_face_in_ipcam_for_saving = False # Flag for saving the frame

            if ipcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_ip = model(resized_ip_for_detection, conf=0.5, verbose=False)
                
                if results_ip and results_ip[0].boxes.data.numel() > 0:
                    for box_data in results_ip[0].boxes:
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                        conf = box_data.conf[0].item()
                        cls = int(box_data.cls[0].item())
                        if cls == FACE_CLASS_ID:
                            found_face_in_ipcam_for_saving = True # Set flag
                            # Draw bounding box on the display frame for IP cam (no counting here yet)
                            cv2.rectangle(ipcam_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Face: {conf:.2f}"
                            cv2.putText(ipcam_display_frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save frame if faces were detected
                if found_face_in_ipcam_for_saving:
                    timestamp = int(time.time() * 1000)
                    frame_filename = os.path.join(output_dir, f"ip_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, frame_ip_original) # Save original high-res frame
                    print(f"Saved IP camera frame: {frame_filename}")
        else:
            print("Warning: IP camera frame read failed. Marking as unavailable.")
            if ip_camera is not None: ip_camera.release()
            ip_camera_available = False
            ipcam_display_frame = None

    # Exit conditions
    if not webcam_available and not ip_camera_available and initial_webcam_intended and initial_ip_camera_intended :
        print("Error: Both cameras are now unavailable. Exiting.")
        break
    elif not webcam_available and initial_webcam_intended and not initial_ip_camera_intended:
        print("Error: Webcam is now unavailable. Exiting.")
        break
    elif not ip_camera_available and initial_ip_camera_intended and not initial_webcam_intended:
        print("Error: IP Camera is now unavailable. Exiting.")
        break

    # Display logic
    display_frame_final = None
    window_title = "Camera Feed"
    display_height, display_width_single = 480, 640
    black_frame = np.zeros((display_height, display_width_single, 3), dtype=np.uint8)
    
    current_webcam_display_safe = webcam_display_frame if webcam_available and webcam_display_frame is not None else black_frame.copy()
    current_ipcam_display_safe = ipcam_display_frame if ip_camera_available and ipcam_display_frame is not None else black_frame.copy()

    if initial_webcam_intended and initial_ip_camera_intended:
        if current_webcam_display_safe.shape[0] != display_height or current_webcam_display_safe.shape[1] != display_width_single:
             current_webcam_display_safe = cv2.resize(current_webcam_display_safe, (display_width_single, display_height))
        if current_ipcam_display_safe.shape[0] != display_height or current_ipcam_display_safe.shape[1] != display_width_single:
             current_ipcam_display_safe = cv2.resize(current_ipcam_display_safe, (display_width_single, display_height))
        display_frame_final = np.hstack((current_webcam_display_safe, current_ipcam_display_safe))
        window_title = 'Camera Feeds (Webcam with Counting | IP Cam)'
    elif initial_webcam_intended:
        display_frame_final = current_webcam_display_safe
        window_title = 'Webcam Feed (Two-Line Person Counting)'
    elif initial_ip_camera_intended:
        display_frame_final = current_ipcam_display_safe
        window_title = 'IP Camera Feed'
    
    if display_frame_final is not None and display_frame_final.size > 0 :
        cv2.imshow(window_title, display_frame_final)
    else:
        if not (initial_webcam_intended or initial_ip_camera_intended):
             cv2.imshow("No Camera Intended", np.zeros((100,100,3), dtype=np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if webcam is not None and webcam.isOpened():
    webcam.release()
if ip_camera is not None and ip_camera.isOpened():
    ip_camera.release()
cv2.destroyAllWindows()
print("Script finished.")