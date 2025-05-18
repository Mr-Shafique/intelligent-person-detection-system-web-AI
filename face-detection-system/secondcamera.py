import cv2
import os
from ultralytics import YOLO
import time
import numpy as np
import math
import pickle
import json
import requests
import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine # For comparing embeddings

# --- Configuration ---
YOLO_MODEL_PATH = 'yolov8n-face.pt'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
OUTPUT_DIR = 'capturedfaces'
DETECTION_LOG_FILE = "detectionlog.json"
RECOGNITION_THRESHOLD = 0.6 # Cosine distance; lower is more similar. Adjust as needed. Max is around 0.6 for SFace being a decent match.
PROCESS_EVERY_N_FRAMES =2 # Process every Nth frame for detection & recognition
FACE_CLASS_ID = 0 # Assuming class 0 is 'face' for yolov8n-face.pt

# --- Person Counting Variables (Webcam & IP Camera) ---
LINE_A_Y = 200
LINE_B_Y = 300
PERSONS_IN_COUNT = 0
PERSONS_OUT_COUNT = 0
IP_PERSONS_IN_COUNT = 0
IP_PERSONS_OUT_COUNT = 0
tracked_faces = {} # {face_id: {'centroid', 'box', 'last_seen_frame', 'crossed_A_pending_B', 'crossed_B_pending_A', 'name', 'cmsId', 'status', 'recognized_in_session'}}
tracked_faces_ip = {}
next_face_id = 0
next_face_id_ip = 0
MAX_FRAMES_UNSEEN = 15 # Increased for potentially slower recognition step
CENTROID_MATCH_THRESHOLD = 75

# --- Load YOLOv8 model ---
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Please ensure '{YOLO_MODEL_PATH}' is in the correct location.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Known Face Embeddings ---
known_face_embeddings = []
def load_known_embeddings(file_path=EMBEDDINGS_FILE):
    global known_face_embeddings
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                known_face_embeddings = pickle.load(f)
            print(f"Successfully loaded {len(known_face_embeddings)} known face embeddings.")
        except Exception as e:
            print(f"Error loading embeddings from {file_path}: {e}")
            known_face_embeddings = []
    else:
        print(f"Warning: Embedding file {file_path} not found. Recognition will not work.")
        known_face_embeddings = []

load_known_embeddings()

# --- Helper Functions for Face Recognition ---
def create_face_embedding_live(face_roi):
    try:
        # DeepFace.represent expects BGR numpy array
        embedding_obj = DeepFace.represent(face_roi,
                                           model_name="Facenet",
                                           enforce_detection=False,
                                           detector_backend='skip',
                                           align=True) # Align helps SFace
        if embedding_obj and len(embedding_obj) > 0 and "embedding" in embedding_obj[0]:
            return embedding_obj[0]["embedding"]
    except Exception as e:
        # print(f"Error generating live embedding: {e}") # Can be verbose
        pass
    return None

def find_match(live_face_roi, known_embeddings_data, threshold):
    live_embedding = create_face_embedding_live(live_face_roi)
    if not live_embedding:
        return "Unknown", None, None, float('inf') # Ensure 4-tuple return

    if not known_embeddings_data: # No known embeddings to compare against
        return "Unknown", None, None, float('inf') # Ensure 4-tuple return

    v1 = np.array(live_embedding)
    if v1.ndim > 1: v1 = v1.flatten()

    # Variables to store the details of the closest face found
    closest_distance = float('inf')
    matched_name = "Unknown"
    matched_cmsId = None
    # This will store the status from the PKL file for the closest face.
    # If 'status' key is missing for that specific entry, it will be None.
    matched_status_from_pkl = None 

    for known_entry in known_embeddings_data:
        known_embedding_vector = known_entry.get("imageEmbedding")
        if not known_embedding_vector:
            # print(f"Warning: Skipping known_entry due to missing 'imageEmbedding': {known_entry.get('name')}")
            continue

        v2 = np.array(known_embedding_vector)
        if v2.ndim > 1: v2 = v2.flatten()

        if v1.shape != v2.shape:
            # print(f"Warning: Skipping embedding due to shape mismatch for {known_entry.get('name')}. Live: {v1.shape}, Known: {v2.shape}")
            continue
        
        try:
            distance = cosine(v1, v2)
        except Exception as e:
            # print(f"Error calculating cosine distance for {known_entry.get('name')}: {e}")
            continue # Skip this entry

        if distance < closest_distance:
            closest_distance = distance
            # This is the new closest face found so far. Store its details.
            matched_name = known_entry.get("name", "Error: Name Missing")
            matched_cmsId = known_entry.get("cmsId", "Error: CMS ID Missing")
            # Get the status directly from the pkl entry.
            # If 'status' key is missing in pkl for this entry, matched_status_from_pkl will be None.
            matched_status_from_pkl = known_entry.get("status") 
                                                
    # After checking all known entries, if the closest one meets the threshold, it's a confident match.
    if closest_distance <= threshold:
        # Return the details of the confident match.
        # matched_status_from_pkl will be the actual status string (e.g., "banned") or None.
        # log_detection_event will handle a None status by converting it to "Unknown".
        return matched_name, matched_cmsId, matched_status_from_pkl, closest_distance
    else:
        # No confident match found, return "Unknown" and None for status.
        return "Unknown", None, None, closest_distance


# --- Helper Function for Logging to Backend ---
BACKEND_API_URL = "http://localhost:5000/api/detections"
def log_detection_event(person_cmsId, person_name, action, camera_source_input, recognized_face_frame_filename=None, status=None): # param camera_source_input is "webcam" or "ipcam"
    
    # Map the input camera source to the desired display name for the log
    camera_source_for_log = camera_source_input # Default to the input if no specific mapping
    if camera_source_input == "webcam":
        camera_source_for_log = "Block 1"
    elif camera_source_input == "ipcam":
        camera_source_for_log = "Block 2"

    event = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "action": action,
        "camera_source": camera_source_for_log, # Use the mapped name here
        "image_saved": recognized_face_frame_filename,
        "person_status_at_event": str(status) if status else "Unknown"
    }
    payload = {
        "person_cmsId": str(person_cmsId) if person_cmsId else "Unknown",
        "person_name": str(person_name) if person_name else "Unknown",
        "status": str(status) if status else "Unknown"  # Always include status, default to "Unknown"
    }

    payload["event"] = event # Add event object last
    
    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=2)
        response.raise_for_status()
        # Update the print statement to use the mapped camera name
        print(f"Logged to backend: {person_name} ({person_cmsId}) - {action} from {camera_source_for_log}")
    except Exception as e:
        print(f"Error sending detection log to backend: {e}")


# --- Camera Initialization ---
webcam = cv2.VideoCapture(0)
ip_camera_url = "http://10.102.138.93:8080/video" # Replace with your IP camera URL
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
print(f"Recognition threshold (cosine distance): {RECOGNITION_THRESHOLD} (lower is stricter)")

if LINE_A_Y >= LINE_B_Y:
    print("Critical Warning: LINE_A_Y should be less than LINE_B_Y for IN=downwards logic.")

webcam_frame_count = 0
ipcam_frame_count = 0
initial_webcam_intended = webcam_available
initial_ip_camera_intended = ip_camera_available

# --- Main Loop ---
while True:
    webcam_display_frame = None
    ipcam_display_frame = None

    # --- Process Webcam (with IN/OUT counting and recognition) ---
    if webcam_available:
        ret_webcam, frame_webcam_original = webcam.read()
        if ret_webcam:
            webcam_frame_count += 1
            resized_webcam_for_detection = cv2.resize(frame_webcam_original, (640, 480))
            scale_x = frame_webcam_original.shape[1] / 640.0
            scale_y = frame_webcam_original.shape[0] / 480.0
            webcam_display_frame = frame_webcam_original.copy() # For drawing

            current_detections_data = [] # Store {'centroid', 'box', 'face_roi'}

            if webcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_webcam = model(resized_webcam_for_detection, conf=0.5, verbose=False) # Lower conf for more detections if needed
                
                if results_webcam and results_webcam[0].boxes.data.numel() > 0:
                    for box_data in results_webcam[0].boxes:
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                        face_roi = frame_webcam_original[y1:y2, x1:x2]
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        current_detections_data.append({
                            'centroid': (cx, cy),
                            'box': (x1, y1, x2, y2),
                            'face_roi': face_roi
                        })
                
                # --- Webcam Tracking & Recognition Logic ---
                temp_current_detections = list(current_detections_data) # Copy for matching

                # Try to match existing tracks
                for face_id, track_data in list(tracked_faces.items()):
                    best_match_idx = -1
                    min_dist_centroid = float('inf')
                    for i, det_data in enumerate(temp_current_detections):
                        dist = math.dist(track_data['centroid'], det_data['centroid'])
                        if dist < CENTROID_MATCH_THRESHOLD and dist < min_dist_centroid:
                            min_dist_centroid = dist
                            best_match_idx = i
                    
                    if best_match_idx != -1: 
                        matched_det_data = temp_current_detections.pop(best_match_idx)
                        prev_cy = track_data['centroid'][1]
                        curr_cy = matched_det_data['centroid'][1]

                        tracked_faces[face_id].update({
                            'centroid': matched_det_data['centroid'],
                            'box': matched_det_data['box'],
                            'last_seen_frame': webcam_frame_count
                        })
                        
                        # --- IN/OUT Counting Logic ---
                        action_taken = None
                        # ... (rest of the IN/OUT logic for A_pending_B, B_pending_A) ...
                        # Check for crossing Line A towards B (potential IN start)
                        if not track_data['crossed_A_pending_B'] and \
                           prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y:
                            tracked_faces[face_id]['crossed_A_pending_B'] = True
                            tracked_faces[face_id]['crossed_B_pending_A'] = False

                        # Check for crossing Line B towards A (potential OUT start)
                        elif not track_data['crossed_B_pending_A'] and \
                             prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y:
                            tracked_faces[face_id]['crossed_B_pending_A'] = True
                            tracked_faces[face_id]['crossed_A_pending_B'] = False

                        # Check for IN completion
                        if track_data['crossed_A_pending_B'] and \
                           prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y:
                            PERSONS_IN_COUNT += 1
                            action_taken = "IN"
                            tracked_faces[face_id]['crossed_A_pending_B'] = False
                            tracked_faces[face_id]['crossed_B_pending_A'] = False

                        # Check for OUT completion
                        elif track_data['crossed_B_pending_A'] and \
                             prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y:
                            PERSONS_OUT_COUNT += 1
                            action_taken = "OUT"
                            tracked_faces[face_id]['crossed_B_pending_A'] = False
                            tracked_faces[face_id]['crossed_A_pending_B'] = False
                        
                        # Reset pending states if person turns back
                        if track_data['crossed_A_pending_B'] and curr_cy < LINE_A_Y:
                            tracked_faces[face_id]['crossed_A_pending_B'] = False
                        if track_data['crossed_B_pending_A'] and curr_cy >= LINE_B_Y:
                            tracked_faces[face_id]['crossed_B_pending_A'] = False

                        if action_taken:
                            print(f"ID {face_id} ({track_data.get('name', 'Unknown')}) COUNTED {action_taken}. Total IN: {PERSONS_IN_COUNT}, Total OUT: {PERSONS_OUT_COUNT}")
                            img_filename_event = None
                            timestamp_img = int(time.time() * 1000)
                            
                            # Determine filename based on whether person is recognized or unknown
                            if track_data.get('cmsId'):
                                img_filename_event = os.path.join(OUTPUT_DIR, f"recognized_{track_data['cmsId']}_{action_taken}_{timestamp_img}.jpg")
                            else: # Person is Unknown
                                img_filename_event = os.path.join(OUTPUT_DIR, f"unknown_webcam_{action_taken}_{timestamp_img}.jpg")
                            
                            # Attempt to save the image
                            try:
                                cv2.imwrite(img_filename_event, frame_webcam_original) 
                                print(f"Saved webcam frame for {action_taken}: {img_filename_event}")
                            except Exception as e:
                                print(f"Error saving webcam frame {img_filename_event}: {e}")
                                img_filename_event = None # Ensure it's None if save failed
                            
                            log_detection_event(track_data.get('cmsId'), track_data.get('name'), action_taken, "webcam", img_filename_event, track_data.get('status'))
                    else: # Track not matched
                        if (webcam_frame_count - track_data['last_seen_frame']) > MAX_FRAMES_UNSEEN:
                            # print(f"Removing lost track ID {face_id} ({track_data.get('name', 'Unknown')})")
                            del tracked_faces[face_id]
                
                # Handle new (unmatched) detections
                for new_det_data in temp_current_detections:
                    name, cmsId, status, dist = "Unknown", None, None, float('inf')
                    if new_det_data['face_roi'] is not None and new_det_data['face_roi'].size > 0 :
                        name, cmsId, status, dist = find_match(new_det_data['face_roi'], known_face_embeddings, RECOGNITION_THRESHOLD)
                    
                    new_id = next_face_id
                    next_face_id += 1
                    tracked_faces[new_id] = {
                        'centroid': new_det_data['centroid'],
                        'box': new_det_data['box'],
                        'last_seen_frame': webcam_frame_count,
                        'crossed_A_pending_B': False,
                        'crossed_B_pending_A': False,
                        'name': name,
                        'cmsId': cmsId,
                        'status': status,
                        'recognized_in_session': True if cmsId else False # Mark if recognition attempt led to a match
                    }
                    # print(f"New track ID {new_id} ({name}, CMS: {cmsId}, Status: {status}, Dist: {dist:.3f}) at {new_det_data['centroid']}")

            # Draw bounding boxes and info for all current tracks on webcam_display_frame
            for face_id, track_data in tracked_faces.items():
                x1, y1, x2, y2 = track_data['box']
                color = (255, 0, 0) if track_data['cmsId'] else (0, 0, 255) # Blue if recognized, Red if not
                cv2.rectangle(webcam_display_frame, (x1, y1), (x2, y2), color, 2)
                
                display_name = track_data.get('name', 'Unknown')
                display_status = track_data.get('status', '')
                label = f"ID:{face_id} {display_name}"
                if display_status and display_name != "Unknown":
                    label += f" ({display_status})"
                
                cv2.putText(webcam_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw counting lines and counts
            cv2.line(webcam_display_frame, (0, LINE_A_Y), (640, LINE_A_Y), (0, 255, 0), 2)
            cv2.putText(webcam_display_frame, "A", (5, LINE_A_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.line(webcam_display_frame, (0, LINE_B_Y), (640, LINE_B_Y), (0, 0, 255), 2)
            cv2.putText(webcam_display_frame, "B", (5, LINE_B_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(webcam_display_frame, f"IN: {PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(webcam_display_frame, f"OUT: {PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else: # Webcam frame read failed
            print("Warning: Webcam frame read failed. Marking as unavailable.")
            if webcam is not None: webcam.release()
            webcam_available = False
            webcam_display_frame = None

    # --- Process IP camera (IN/OUT counting and recognition) ---
    if ip_camera_available:
        ret_ip, frame_ip_original = ip_camera.read()
        if ret_ip:
            ipcam_frame_count += 1
            resized_ip_for_detection = cv2.resize(frame_ip_original, (640, 480))
            ipcam_display_frame = resized_ip_for_detection.copy() # For drawing
            current_detections_data_ip = []
            if ipcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_ip = model(resized_ip_for_detection, conf=0.5, verbose=False)
                if results_ip and results_ip[0].boxes.data.numel() > 0:
                    for box_data in results_ip[0].boxes:
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                        conf_yolo = box_data.conf[0].item()
                        cls = int(box_data.cls[0].item())
                        if cls == FACE_CLASS_ID:
                            y1_crop, y2_crop = max(0, y1), min(resized_ip_for_detection.shape[0], y2)
                            x1_crop, x2_crop = max(0, x1), min(resized_ip_for_detection.shape[1], x2)
                            face_roi_ip = None
                            if y1_crop < y2_crop and x1_crop < x2_crop:
                                face_roi_ip = resized_ip_for_detection[y1_crop:y2_crop, x1_crop:x2_crop]
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            current_detections_data_ip.append({
                                'centroid': (cx, cy),
                                'box': (x1, y1, x2, y2),
                                'face_roi': face_roi_ip
                            })
            temp_current_detections_ip = list(current_detections_data_ip)
            for face_id, track_data in list(tracked_faces_ip.items()):
                best_match_idx = -1
                min_dist_centroid = float('inf')
                for i, det_data in enumerate(temp_current_detections_ip):
                    dist = math.dist(track_data['centroid'], det_data['centroid'])
                    if dist < CENTROID_MATCH_THRESHOLD and dist < min_dist_centroid:
                        min_dist_centroid = dist
                        best_match_idx = i
                if best_match_idx != -1:
                    matched_det_data = temp_current_detections_ip.pop(best_match_idx)
                    prev_cy = track_data['centroid'][1]
                    curr_cy = matched_det_data['centroid'][1]
                    tracked_faces_ip[face_id].update({
                        'centroid': matched_det_data['centroid'],
                        'box': matched_det_data['box'],
                        'last_seen_frame': ipcam_frame_count
                    })
                    action_taken = None
                    if not track_data['crossed_A_pending_B'] and prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y:
                        tracked_faces_ip[face_id]['crossed_A_pending_B'] = True
                        tracked_faces_ip[face_id]['crossed_B_pending_A'] = False
                    elif not track_data['crossed_B_pending_A'] and prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y:
                        tracked_faces_ip[face_id]['crossed_B_pending_A'] = True
                        tracked_faces_ip[face_id]['crossed_A_pending_B'] = False
                    if track_data['crossed_A_pending_B'] and prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y:
                        IP_PERSONS_IN_COUNT += 1
                        action_taken = "IN"
                        tracked_faces_ip[face_id]['crossed_A_pending_B'] = False
                        tracked_faces_ip[face_id]['crossed_B_pending_A'] = False
                    elif track_data['crossed_B_pending_A'] and prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y:
                        IP_PERSONS_OUT_COUNT += 1
                        action_taken = "OUT"
                        tracked_faces_ip[face_id]['crossed_B_pending_A'] = False
                        tracked_faces_ip[face_id]['crossed_A_pending_B'] = False
                    # Reset pending states if person turns back
                    if track_data['crossed_A_pending_B'] and curr_cy < LINE_A_Y:
                        tracked_faces_ip[face_id]['crossed_A_pending_B'] = False
                    if track_data['crossed_B_pending_A'] and curr_cy >= LINE_B_Y:
                        tracked_faces_ip[face_id]['crossed_B_pending_A'] = False
                    if action_taken:
                        print(f"[IPCAM] ID {face_id} ({track_data.get('name', 'Unknown')}) COUNTED {action_taken}. Total IN: {IP_PERSONS_IN_COUNT}, Total OUT: {IP_PERSONS_OUT_COUNT}")
                        img_filename_event = None
                        timestamp_img = int(time.time() * 1000)

                        # Determine filename based on whether person is recognized or unknown
                        if track_data.get('cmsId'):
                            img_filename_event = os.path.join(OUTPUT_DIR, f"ipcam_recognized_{track_data['cmsId']}_{action_taken}_{timestamp_img}.jpg")
                        else: # Person is Unknown
                            img_filename_event = os.path.join(OUTPUT_DIR, f"ipcam_unknown_{action_taken}_{timestamp_img}.jpg")
                        
                        # Attempt to save the image (using frame_ip_original for better quality)
                        try:
                            cv2.imwrite(img_filename_event, frame_ip_original) 
                            print(f"Saved IP cam frame for {action_taken}: {img_filename_event}")
                        except Exception as e:
                            print(f"Error saving IP cam frame {img_filename_event}: {e}")
                            img_filename_event = None # Ensure it's None if save failed
                            
                        log_detection_event(track_data.get('cmsId'), track_data.get('name'), action_taken, "ipcam", img_filename_event, track_data.get('status'))
                else: # Track not matched
                    if (ipcam_frame_count - track_data['last_seen_frame']) > MAX_FRAMES_UNSEEN:
                        del tracked_faces_ip[face_id]
            for new_det_data in temp_current_detections_ip:
                name, cmsId, status, dist = "Unknown", None, None, float('inf')
                if new_det_data['face_roi'] is not None and new_det_data['face_roi'].size > 0:
                    name, cmsId, status, dist = find_match(new_det_data['face_roi'], known_face_embeddings, RECOGNITION_THRESHOLD)
                new_id = next_face_id_ip
                next_face_id_ip += 1
                tracked_faces_ip[new_id] = {
                    'centroid': new_det_data['centroid'],
                    'box': new_det_data['box'],
                    'last_seen_frame': ipcam_frame_count,
                    'crossed_A_pending_B': False,
                    'crossed_B_pending_A': False,
                    'name': name,
                    'cmsId': cmsId,
                    'status': status,
                    'recognized_in_session': True if cmsId else False
                }
            for face_id, track_data in tracked_faces_ip.items():
                x1, y1, x2, y2 = track_data['box']
                color = (255, 0, 0) if track_data['cmsId'] else (0, 0, 255)
                cv2.rectangle(ipcam_display_frame, (x1, y1), (x2, y2), color, 2)
                display_name = track_data.get('name', 'Unknown')
                display_status = track_data.get('status', '')
                label = f"ID:{face_id} {display_name}"
                if display_status and display_name != "Unknown":
                    label += f" ({display_status})"
                cv2.putText(ipcam_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.line(ipcam_display_frame, (0, LINE_A_Y), (640, LINE_A_Y), (0, 255, 0), 2)
            cv2.putText(ipcam_display_frame, "A", (5, LINE_A_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.line(ipcam_display_frame, (0, LINE_B_Y), (640, LINE_B_Y), (0, 0, 255), 2)
            cv2.putText(ipcam_display_frame, "B", (5, LINE_B_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(ipcam_display_frame, f"IN: {IP_PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(ipcam_display_frame, f"OUT: {IP_PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            print("Warning: IP camera frame read failed. Marking as unavailable.")
            if ip_camera is not None: ip_camera.release()
            ip_camera_available = False
            ipcam_display_frame = None

    # --- Display Logic & Exit Conditions ---
    # (This part remains largely the same as your original, handling display of one or both frames)
    if not webcam_available and not ip_camera_available and initial_webcam_intended and initial_ip_camera_intended :
        print("Error: Both cameras are now unavailable. Exiting.")
        break
    # ... (other exit conditions for single camera failures)

    display_frame_final = None
    window_title = "Camera Feed"
    display_height, display_width_single = 480, 640
    black_frame = np.zeros((display_height, display_width_single, 3), dtype=np.uint8)
    
    current_webcam_display_safe = webcam_display_frame if webcam_available and webcam_display_frame is not None else black_frame.copy()
    current_ipcam_display_safe = ipcam_display_frame if ip_camera_available and ipcam_display_frame is not None else black_frame.copy()

    if current_webcam_display_safe.shape[0] != display_height or current_webcam_display_safe.shape[1] != display_width_single:
            current_webcam_display_safe = cv2.resize(current_webcam_display_safe, (display_width_single, display_height))
    if current_ipcam_display_safe.shape[0] != display_height or current_ipcam_display_safe.shape[1] != display_width_single:
            current_ipcam_display_safe = cv2.resize(current_ipcam_display_safe, (display_width_single, display_height))

    if initial_webcam_intended and initial_ip_camera_intended:
        display_frame_final = np.hstack((current_webcam_display_safe, current_ipcam_display_safe))
        window_title = 'Webcam (Counting & Reco) | IP Cam (Reco)'
    elif initial_webcam_intended:
        display_frame_final = current_webcam_display_safe
        window_title = 'Webcam Feed (Counting & Recognition)'
    elif initial_ip_camera_intended:
        display_frame_final = current_ipcam_display_safe
        window_title = 'IP Camera Feed (Recognition)'
    
    if display_frame_final is not None and display_frame_final.size > 0 :
        cv2.imshow(window_title, display_frame_final)
    else:
        # This case might occur if both cams fail but one wasn't initially intended,
        # or if no camera was intended from the start.
        if not (initial_webcam_intended or initial_ip_camera_intended):
            pass # No window if no camera was ever intended/available
        elif not (webcam_available or ip_camera_available): # Both failed but at least one was intended
             cv2.imshow("All Cameras Failed", np.zeros((100,300,3), dtype=np.uint8))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
if webcam is not None and webcam.isOpened():
    webcam.release()
if ip_camera is not None and ip_camera.isOpened():
    ip_camera.release()
cv2.destroyAllWindows()
print("Script finished.")