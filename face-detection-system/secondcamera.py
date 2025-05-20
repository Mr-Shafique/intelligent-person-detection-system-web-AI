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
from flask import Flask, Response
import threading
from queue import Queue as ThreadQueue # For thread-safe frame passing (renamed to avoid conflict if user defines another Queue)

# --- Configuration ---
YOLO_MODEL_PATH = 'yolov8n-face.pt'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
OUTPUT_DIR = 'capturedfaces'
DETECTION_LOG_FILE = "detectionlog.json" # Not actively used for writing in this version, but kept for reference
RECOGNITION_THRESHOLD = 0.4 # Cosine distance; lower is more similar.
RECOGNITION_RE_MATCH_THRESHOLD = 0.3 # Slightly more lenient for re-matching from recently_lost_cache
PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame for detection & recognition. HIGHER = SMOOTHER VISUALS, LESS FREQUENT UPDATES.
FACE_CLASS_ID = 0 # Assuming class 0 is 'face' for yolov8n-face.pt
RECENTLY_LOST_TIMEOUT_SECONDS = 10 # How long to keep a face in the 'recently_lost' cache

# --- Thread-safe Camera Stream Class ---
class CameraStream:
    def __init__(self, src=0, name="CameraStream"):
        self.stream = None # Initialize to None
        try:
            self.stream = cv2.VideoCapture(src)
            if not self.stream or not self.stream.isOpened():
                print(f"Warning: Could not open camera {src}")
                self.grabbed = False
                self.frame = None
                self.running = False
                return
        except Exception as e:
            print(f"Exception opening camera {src}: {e}")
            self.grabbed = False
            self.frame = None
            self.running = False
            return


        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
             print(f"Warning: {self.name} - initial frame read failed.")
             self.running = False
             if self.stream.isOpened(): self.stream.release()
             return

        self.name = name
        self.running = True
        self.thread = threading.Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"{self.name} started.")

    def update(self):
        while self.running:
            if self.stream and self.stream.isOpened():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    print(f"Warning: {self.name} - frame read failed or stream ended.")
                    self.running = False
                    break
                self.frame = frame
            else:
                print(f"Warning: {self.name} - stream is not open in update loop.")
                self.running = False
                break
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        print(f"Stopping {self.name}...")
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.stream and self.stream.isOpened():
            self.stream.release()
        print(f"{self.name} stopped and released.")

    def isOpened(self):
        return self.stream is not None and self.stream.isOpened() and self.running


# --- Person Counting Variables (Webcam & IP Camera) ---
LINE_A_Y = 200
LINE_B_Y = 300
PERSONS_IN_COUNT = 0 # For Camera 0 (formerly webcam)
PERSONS_OUT_COUNT = 0 # For Camera 0
CAM2_PERSONS_IN_COUNT = 0 # For Camera 2 (formerly IP Camera)
CAM2_PERSONS_OUT_COUNT = 0 # For Camera 2

tracked_faces = {} # For Camera 0 {face_id: {'centroid', 'box', 'last_seen_frame_count', 'crossed_A_pending_B', 'crossed_B_pending_A', 'name', 'cmsId', 'status', 'live_embedding', 'recognized_in_session'}}
tracked_faces_cam2 = {} # For Camera 2
next_face_id = 0 # For Camera 0
next_face_id_cam2 = 0 # For Camera 2

recently_lost_faces_webcam = {} # For Camera 0
recently_lost_faces_cam2 = {} # For Camera 2

MAX_FRAMES_UNSEEN_PROCESSING_CYCLES = 3 # No. of *processing cycles* a track can be unseen. e.g. if PROCESS_EVERY_N_FRAMES=5, this means 3*5=15 actual frames.
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
        print(f"Warning: Embedding file {file_path} not found. Recognition will not work well for new faces.")
        known_face_embeddings = []

load_known_embeddings()

# --- Helper Functions for Face Recognition ---
def create_face_embedding_live(face_roi):
    if face_roi is None or face_roi.size == 0:
        return None
    try:
        embedding_obj = DeepFace.represent(face_roi,
                                           model_name="Facenet", # Consider "SFace" for speed
                                           enforce_detection=False,
                                           detector_backend='skip',
                                           align=True)
        if embedding_obj and len(embedding_obj) > 0 and "embedding" in embedding_obj[0]:
            return embedding_obj[0]["embedding"]
    except Exception as e:
        # print(f"Error generating live embedding: {e}") # Can be verbose
        pass
    return None

def find_match_against_known_db(live_embedding, known_embeddings_data, threshold):
    if not live_embedding:
        return "Unknown", None, None, float('inf')
    if not known_embeddings_data:
        return "Unknown", None, None, float('inf')

    v1 = np.array(live_embedding).flatten()
    closest_distance = float('inf')
    matched_name = "Unknown"
    matched_cmsId = None
    matched_status_from_pkl = None

    for known_entry in known_embeddings_data:
        known_embedding_vector = known_entry.get("imageEmbedding")
        if not known_embedding_vector:
            continue
        v2 = np.array(known_embedding_vector).flatten()
        if v1.shape != v2.shape:
            continue
        try:
            distance = cosine(v1, v2)
        except Exception:
            continue
        if distance < closest_distance:
            closest_distance = distance
            matched_name = known_entry.get("name", "Error: Name Missing")
            matched_cmsId = known_entry.get("cmsId", "Error: CMS ID Missing")
            matched_status_from_pkl = known_entry.get("status")

    if closest_distance <= threshold:
        return matched_name, matched_cmsId, matched_status_from_pkl, closest_distance
    else:
        return "Unknown", None, None, closest_distance

# --- Helper Function for Logging to Backend ---
BACKEND_API_URL = "http://localhost:5000/api/detections"
def log_detection_event(person_cmsId, person_name, action, camera_source_input, recognized_face_frame_filename=None, status=None):
    # camera_source_input will now be "camera0" or "camera2"
    camera_source_for_log = "Block 1" # Changed to always log as Block 1
    event = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "action": action,
        "camera_source": camera_source_for_log,
        "image_saved": recognized_face_frame_filename,
        "person_status_at_event": str(status) if status else "Unknown"
    }
    payload = {
        "person_cmsId": str(person_cmsId) if person_cmsId else "Unknown",
        "person_name": str(person_name) if person_name else "Unknown",
        "status": str(status) if status else "Unknown",
        "event": event
    }
    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=2)
        response.raise_for_status()
        print(f"Logged to backend: {person_name} ({person_cmsId}) - {action} from {camera_source_for_log}")
    except Exception as e:
        print(f"Error sending detection log to backend: {e}")

def save_and_log_event_async(frame_to_save, output_dir, base_filename_prefix, action_taken, cms_id, person_name, camera_source, person_status, event_timestamp):
    """
    Saves an image and logs the event asynchronously.
    """
    img_fn = None
    if frame_to_save is not None:
        filename_ts = int(event_timestamp * 1000)
        img_fn = os.path.join(output_dir, f"{base_filename_prefix}_{action_taken}_{filename_ts}.jpg")
        try:
            cv2.imwrite(img_fn, frame_to_save)
            print(f"Saved: {img_fn}")
        except Exception as e:
            print(f"Error saving image {img_fn}: {e}")
            img_fn = None # Ensure img_fn is None if saving failed
    
    # Log event regardless of image saving success, but pass actual img_fn
    log_detection_event(cms_id, person_name, action_taken, camera_source, img_fn, person_status)

# --- Camera Initialization ---
webcam_src = 0 # Camera 0 index
camera2_src = 2 # Camera 2 index

webcam_stream = CameraStream(src=webcam_src, name="Camera0Stream")
camera2_stream = CameraStream(src=camera2_src, name="Camera2Stream") # Changed from IP Camera URL to index 2

webcam_available = webcam_stream.isOpened()
camera2_available = camera2_stream.isOpened() # Changed from ip_camera_available

if not webcam_available and not camera2_available: # Changed from ip_camera_available
    print("Error: Neither Camera 0 nor Camera 2 is available. Exiting.")
    if webcam_stream: webcam_stream.stop()
    if camera2_stream: camera2_stream.stop() # Changed from ip_camera_stream
    cv2.destroyAllWindows()
    exit()

print("Press 'q' to quit.")
print(f"Recognition threshold (cosine distance): {RECOGNITION_THRESHOLD} (lower is stricter)")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames for AI tasks.")

if LINE_A_Y >= LINE_B_Y:
    print("Critical Warning: LINE_A_Y should be less than LINE_B_Y for IN=downwards logic.")

webcam_frame_counter = 0 # Renamed to avoid conflict with cv2.CAP_PROP_FRAME_COUNT
cam2_frame_counter = 0  # Renamed from ipcam_frame_counter

initial_webcam_intended = webcam_available
initial_camera2_intended = camera2_available # Renamed from initial_ip_camera_intended

# --- Flask App for Streaming ---
app = Flask(__name__)
latest_frame_flask = None # Renamed to avoid confusion with other 'frame' variables
frame_lock_flask = threading.Lock()

def update_latest_frame_flask(frame):
    global latest_frame_flask
    with frame_lock_flask:
        latest_frame_flask = frame.copy() if frame is not None else None

def gen_frames_flask():
    global latest_frame_flask
    while True:
        with frame_lock_flask:
            frame_to_encode = latest_frame_flask.copy() if latest_frame_flask is not None else None
        if frame_to_encode is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_encode)
            if not ret:
                time.sleep(0.01) # Avoid busy loop on encoding failure
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05) # Wait if no frame is available

@app.route('/video_feed')
def video_feed_flask():
    return Response(gen_frames_flask(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    print("Starting Flask server on 0.0.0.0:5002")
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# --- Main Loop ---
processing_frame_count_webcam = 0 # Counts frames on which processing is done
processing_frame_count_cam2 = 0   # Renamed from processing_frame_count_ipcam

while True:
    webcam_display_frame = None
    cam2_display_frame = None # Renamed from ipcam_display_frame
    frame_webcam_original = None
    frame_cam2_original = None # Renamed from frame_ip_original

    # --- Process Webcam (Camera 0) ---
    if webcam_available:
        frame_webcam_original = webcam_stream.read()
        if frame_webcam_original is not None:
            webcam_frame_counter += 1
            webcam_display_frame = frame_webcam_original.copy() # Always have a display frame

            if webcam_frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                processing_frame_count_webcam +=1
                resized_webcam_for_detection = cv2.resize(frame_webcam_original, (640, 480))
                scale_x = frame_webcam_original.shape[1] / 640.0
                scale_y = frame_webcam_original.shape[0] / 480.0
                
                current_detections_data = []
                results_webcam = model(resized_webcam_for_detection, conf=0.5, verbose=False)
                if results_webcam and results_webcam[0].boxes.data.numel() > 0:
                    for box_data in results_webcam[0].boxes:
                        x1_r, y1_r, x2_r, y2_r = map(int, box_data.xyxy[0]) # Coords on resized
                        x1_o, y1_o = int(x1_r * scale_x), int(y1_r * scale_y) # Coords on original
                        x2_o, y2_o = int(x2_r * scale_x), int(y2_r * scale_y)
                        
                        face_roi_original = None
                        if y1_o < y2_o and x1_o < x2_o: # Ensure valid ROI
                             face_roi_original = frame_webcam_original[max(0,y1_o):min(frame_webcam_original.shape[0],y2_o), 
                                                                       max(0,x1_o):min(frame_webcam_original.shape[1],x2_o)]
                        
                        cx_o = (x1_o + x2_o) // 2
                        cy_o = (y1_o + y2_o) // 2
                        current_detections_data.append({
                            'centroid': (cx_o, cy_o),       # On original frame
                            'box': (x1_o, y1_o, x2_o, y2_o), # On original frame
                            'face_roi_original': face_roi_original
                        })
                
                matched_track_ids_this_cycle = set()
                # Match current detections to existing tracks
                for det_idx, det_data in enumerate(current_detections_data):
                    best_match_id = -1
                    min_dist_centroid = CENTROID_MATCH_THRESHOLD 
                    for face_id, track_data in tracked_faces.items():
                        dist = math.dist(track_data['centroid'], det_data['centroid'])
                        if dist < min_dist_centroid:
                            min_dist_centroid = dist
                            best_match_id = face_id
                    
                    if best_match_id != -1:
                        matched_track_ids_this_cycle.add(best_match_id)
                        track = tracked_faces[best_match_id]
                        prev_cy = track['centroid'][1]
                        curr_cy = det_data['centroid'][1]
                        track.update({
                            'centroid': det_data['centroid'],
                            'box': det_data['box'],
                            'last_seen_frame_count': processing_frame_count_webcam # Use processing cycle counter
                        })
                        action_taken = None
                        # ... IN/OUT logic ...
                        if not track['crossed_A_pending_B'] and prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y: track['crossed_A_pending_B'] = True; track['crossed_B_pending_A'] = False
                        elif not track['crossed_B_pending_A'] and prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y: track['crossed_B_pending_A'] = True; track['crossed_A_pending_B'] = False
                        if track['crossed_A_pending_B'] and prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y: PERSONS_IN_COUNT += 1; action_taken = "IN"; track['crossed_A_pending_B'] = False
                        elif track['crossed_B_pending_A'] and prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y: PERSONS_OUT_COUNT += 1; action_taken = "OUT"; track['crossed_B_pending_A'] = False
                        if track['crossed_A_pending_B'] and curr_cy < LINE_A_Y: track['crossed_A_pending_B'] = False
                        if track['crossed_B_pending_A'] and curr_cy >= LINE_B_Y: track['crossed_B_pending_A'] = False
                        
                        if action_taken:
                            event_time = time.time()
                            print(f"ID {best_match_id} ({track.get('name', 'Unknown')}) COUNTED {action_taken}. Total IN: {PERSONS_IN_COUNT}, Total OUT: {PERSONS_OUT_COUNT}")
                            
                            frame_copy_for_saving = frame_webcam_original.copy() if frame_webcam_original is not None else None
                            filename_prefix = f"{('recognized_cam0_' + str(track['cmsId'])) if track.get('cmsId') else 'unknown_camera0'}"
                            
                            threading.Thread(target=save_and_log_event_async, args=(
                                frame_copy_for_saving, OUTPUT_DIR, filename_prefix, action_taken,
                                track.get('cmsId'), track.get('name'), "camera0", track.get('status'), event_time
                            )).start()
                        current_detections_data[det_idx] = None # Mark as matched
                
                current_detections_data = [d for d in current_detections_data if d is not None] # Filter out matched

                # Handle lost tracks
                lost_ids = []
                for face_id, track_data in tracked_faces.items():
                    if face_id not in matched_track_ids_this_cycle:
                        if (processing_frame_count_webcam - track_data['last_seen_frame_count']) > MAX_FRAMES_UNSEEN_PROCESSING_CYCLES:
                            lost_ids.append(face_id)
                            if track_data.get('cmsId') and track_data.get('live_embedding'):
                                print(f"Webcam: Moving recognized {track_data['name']} (ID {face_id}) to lost cache.")
                                recently_lost_faces_webcam[face_id] = {
                                    'name': track_data['name'], 'cmsId': track_data['cmsId'], 
                                    'status': track_data['status'], 'last_embedding': track_data['live_embedding'],
                                    'lost_timestamp': time.time(), 'last_box': track_data['box']
                                }
                for face_id in lost_ids: del tracked_faces[face_id]

                # Handle new (unmatched by existing tracks) detections
                current_time = time.time() # For cache cleanup
                for lost_id, lost_data in list(recently_lost_faces_webcam.items()): # Cleanup cache
                    if current_time - lost_data['lost_timestamp'] > RECENTLY_LOST_TIMEOUT_SECONDS:
                        del recently_lost_faces_webcam[lost_id]
                
                for new_det_data in current_detections_data:
                    name, cmsId, status, dist = "Unknown", None, None, float('inf')
                    live_emb = create_face_embedding_live(new_det_data['face_roi_original'])
                    revived_id = None

                    if live_emb:
                        # Try to match with recently lost
                        best_lost_match_val = float('inf')
                        for lost_id, lost_data in recently_lost_faces_webcam.items():
                            dist_to_lost = cosine(np.array(live_emb).flatten(), np.array(lost_data['last_embedding']).flatten())
                            if dist_to_lost < RECOGNITION_RE_MATCH_THRESHOLD and dist_to_lost < best_lost_match_val:
                                best_lost_match_val = dist_to_lost
                                revived_id = lost_id
                        
                        if revived_id:
                            matched_lost_data = recently_lost_faces_webcam[revived_id]
                            name, cmsId, status, dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val
                            print(f"Webcam: Re-ID {name} from lost cache. Dist: {dist:.3f}")
                            del recently_lost_faces_webcam[revived_id]
                        else: # Not in lost cache, try full DB
                            name, cmsId, status, dist = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                    
                    new_id = next_face_id
                    next_face_id += 1
                    tracked_faces[new_id] = {
                        'centroid': new_det_data['centroid'], 'box': new_det_data['box'],
                        'last_seen_frame_count': processing_frame_count_webcam,
                        'crossed_A_pending_B': False, 'crossed_B_pending_A': False,
                        'name': name, 'cmsId': cmsId, 'status': status,
                        'live_embedding': live_emb if cmsId else None, # Store if recognized
                        'recognized_in_session': bool(cmsId)
                    }
                    # print(f"Webcam: New track ID {new_id} ({name}, CMS: {cmsId}, Status: {status}, Dist: {dist:.3f})")

            # Draw on webcam_display_frame (always, using last known track data)
            for face_id, track_data in tracked_faces.items():
                x1, y1, x2, y2 = track_data['box']
                color = (255, 0, 0) if track_data['cmsId'] else (0, 0, 255)
                cv2.rectangle(webcam_display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{face_id} {track_data.get('name', 'Unk')}"
                if track_data.get('status') and track_data.get('name', 'Unk') != 'Unknown': label += f" ({track_data.get('status')})"
                cv2.putText(webcam_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.putText(webcam_display_frame, f"IN: {PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(webcam_display_frame, f"OUT: {PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            webcam_stream.stop(); webcam_available = False; webcam_display_frame = None

    # --- Process Camera 2 (formerly IP camera) ---
    if camera2_available: # Renamed from ip_camera_available
        frame_cam2_original = camera2_stream.read() # Renamed from frame_ip_original, camera2_stream
        if frame_cam2_original is not None:
            cam2_frame_counter += 1 # Renamed from ipcam_frame_counter
            cam2_display_frame = frame_cam2_original.copy() # Display original resolution, Renamed from ipcam_display_frame

            if cam2_frame_counter % PROCESS_EVERY_N_FRAMES == 0: # Renamed from ipcam_frame_counter
                processing_frame_count_cam2 +=1 # Renamed from processing_frame_count_ipcam
                
                # Resize for detection, similar to Camera 0
                resized_cam2_for_detection = cv2.resize(frame_cam2_original, (640, 480)) # Renamed
                scale_x_cam2 = frame_cam2_original.shape[1] / 640.0 # Renamed
                scale_y_cam2 = frame_cam2_original.shape[0] / 480.0 # Renamed

                current_detections_data_cam2 = [] # Renamed from current_detections_data_ip
                results_cam2 = model(resized_cam2_for_detection, conf=0.5, verbose=False) # Renamed from results_ip
                if results_cam2 and results_cam2[0].boxes.data.numel() > 0: # Renamed
                    for box_data in results_cam2[0].boxes: # Renamed
                        if int(box_data.cls[0].item()) == FACE_CLASS_ID:
                            x1_r, y1_r, x2_r, y2_r = map(int, box_data.xyxy[0]) # Coords on resized

                            # Scale to original frame coordinates
                            x1_o, y1_o = int(x1_r * scale_x_cam2), int(y1_r * scale_y_cam2) # Use new scales
                            x2_o, y2_o = int(x2_r * scale_x_cam2), int(y2_r * scale_y_cam2)
                            
                            face_roi_original_cam2 = None # Renamed
                            if y1_o < y2_o and x1_o < x2_o:
                                face_roi_original_cam2 = frame_cam2_original[max(0,y1_o):min(frame_cam2_original.shape[0],y2_o), 
                                                                             max(0,x1_o):min(frame_cam2_original.shape[1],x2_o)]

                            cx_o = (x1_o + x2_o) // 2 # Centroid on original
                            cy_o = (y1_o + y2_o) // 2
                            current_detections_data_cam2.append({ # Renamed
                                'centroid': (cx_o, cy_o), # On original frame
                                'box': (x1_o, y1_o, x2_o, y2_o), # On original frame
                                'face_roi_original': face_roi_original_cam2 # Renamed
                            })
                
                matched_track_ids_cam2_this_cycle = set() # Renamed
                # Match current detections to existing Camera 2 tracks
                for det_idx, det_data in enumerate(current_detections_data_cam2): # Renamed
                    best_match_id = -1
                    min_dist_centroid = CENTROID_MATCH_THRESHOLD
                    for face_id, track_data in tracked_faces_cam2.items(): # Renamed from tracked_faces_ip
                        dist = math.dist(track_data['centroid'], det_data['centroid'])
                        if dist < min_dist_centroid:
                            min_dist_centroid = dist
                            best_match_id = face_id
                    
                    if best_match_id != -1:
                        matched_track_ids_cam2_this_cycle.add(best_match_id) # Renamed
                        track = tracked_faces_cam2[best_match_id] # Renamed
                        prev_cy = track['centroid'][1]
                        curr_cy = det_data['centroid'][1]
                        track.update({
                            'centroid': det_data['centroid'], # Original coords
                            'box': det_data['box'], # Original coords, Renamed from 'box_resized'
                            'last_seen_frame_count': processing_frame_count_cam2 # Renamed
                        })
                        action_taken = None
                        # Camera 2 IN/OUT (opposite to Camera 0: A->B is OUT, B->A is IN)
                        if not track['crossed_A_pending_B'] and prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y: track['crossed_A_pending_B'] = True; track['crossed_B_pending_A'] = False
                        elif not track['crossed_B_pending_A'] and prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y: track['crossed_B_pending_A'] = True; track['crossed_A_pending_B'] = False
                        if track['crossed_A_pending_B'] and prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y: CAM2_PERSONS_OUT_COUNT += 1; action_taken = "OUT"; track['crossed_A_pending_B'] = False # Renamed count
                        elif track['crossed_B_pending_A'] and prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y: CAM2_PERSONS_IN_COUNT += 1; action_taken = "IN"; track['crossed_B_pending_A'] = False    # Renamed count
                        if track['crossed_A_pending_B'] and curr_cy < LINE_A_Y: track['crossed_A_pending_B'] = False
                        if track['crossed_B_pending_A'] and curr_cy >= LINE_B_Y: track['crossed_B_pending_A'] = False

                        if action_taken:
                            event_time_cam2 = time.time() # Renamed
                            print(f"[CAM2] ID {best_match_id} ({track.get('name', 'Unknown')}) COUNTED {action_taken}. Total IN: {CAM2_PERSONS_IN_COUNT}, Total OUT: {CAM2_PERSONS_OUT_COUNT}") # Renamed counts
                            
                            frame_copy_for_saving_cam2 = frame_cam2_original.copy() if frame_cam2_original is not None else None # Renamed
                            filename_prefix_cam2 = f"{('recognized_cam2_' + str(track['cmsId'])) if track.get('cmsId') else 'unknown_camera2'}" # Renamed

                            threading.Thread(target=save_and_log_event_async, args=(
                                frame_copy_for_saving_cam2, OUTPUT_DIR, filename_prefix_cam2, action_taken,
                                track.get('cmsId'), track.get('name'), "camera2", track.get('status'), event_time_cam2 # Pass "camera2"
                            )).start()
                        current_detections_data_cam2[det_idx] = None # Renamed
                
                current_detections_data_cam2 = [d for d in current_detections_data_cam2 if d is not None] # Renamed

                # Handle lost Camera 2 tracks
                lost_ids_cam2 = [] # Renamed
                for face_id, track_data in tracked_faces_cam2.items(): # Renamed
                    if face_id not in matched_track_ids_cam2_this_cycle: # Renamed
                        if (processing_frame_count_cam2 - track_data['last_seen_frame_count']) > MAX_FRAMES_UNSEEN_PROCESSING_CYCLES: # Renamed
                            lost_ids_cam2.append(face_id) # Renamed
                            if track_data.get('cmsId') and track_data.get('live_embedding'):
                                print(f"Camera2: Moving recognized {track_data['name']} (ID {face_id}) to lost cache.")
                                recently_lost_faces_cam2[face_id] = { # Renamed
                                    'name': track_data['name'], 'cmsId': track_data['cmsId'],
                                    'status': track_data['status'], 'last_embedding': track_data['live_embedding'],
                                    'lost_timestamp': time.time(), 'last_box': track_data['box'] # Use 'box' (original coords)
                                }
                for face_id in lost_ids_cam2: del tracked_faces_cam2[face_id] # Renamed

                # Handle new Camera 2 detections
                current_time_cam2 = time.time() # Renamed
                for lost_id, lost_data in list(recently_lost_faces_cam2.items()): # Cleanup cache, Renamed
                    if current_time_cam2 - lost_data['lost_timestamp'] > RECENTLY_LOST_TIMEOUT_SECONDS: # Renamed
                        del recently_lost_faces_cam2[lost_id] # Renamed

                for new_det_data in current_detections_data_cam2: # Renamed
                    name, cmsId, status, dist = "Unknown", None, None, float('inf')
                    live_emb = create_face_embedding_live(new_det_data['face_roi_original'])
                    revived_id_cam2 = None # Renamed

                    if live_emb:
                        best_lost_match_val_cam2 = float('inf') # Renamed
                        for lost_id, lost_data in recently_lost_faces_cam2.items(): # Renamed
                            dist_to_lost = cosine(np.array(live_emb).flatten(), np.array(lost_data['last_embedding']).flatten())
                            if dist_to_lost < RECOGNITION_RE_MATCH_THRESHOLD and dist_to_lost < best_lost_match_val_cam2: # Renamed
                                best_lost_match_val_cam2 = dist_to_lost # Renamed
                                revived_id_cam2 = lost_id # Renamed
                        
                        if revived_id_cam2: # Renamed
                            matched_lost_data = recently_lost_faces_cam2[revived_id_cam2] # Renamed
                            name, cmsId, status, dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val_cam2 # Renamed
                            print(f"Camera2: Re-ID {name} from lost cache. Dist: {dist:.3f}")
                            del recently_lost_faces_cam2[revived_id_cam2] # Renamed
                        else:
                            name, cmsId, status, dist = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                    
                    new_id = next_face_id_cam2 # Renamed
                    next_face_id_cam2 += 1 # Renamed
                    tracked_faces_cam2[new_id] = { # Renamed
                        'centroid': new_det_data['centroid'], 'box': new_det_data['box'], # Use 'box' (original coords)
                        'last_seen_frame_count': processing_frame_count_cam2, # Renamed
                        'crossed_A_pending_B': False, 'crossed_B_pending_A': False,
                        'name': name, 'cmsId': cmsId, 'status': status,
                        'live_embedding': live_emb if cmsId else None,
                        'recognized_in_session': bool(cmsId)
                    }
                    # print(f"Camera2: New track ID {new_id} ({name}, CMS: {cmsId}, Status: {status}, Dist: {dist:.3f})")
            
            # Draw on cam2_display_frame (always, using original coordinates)
            for face_id, track_data in tracked_faces_cam2.items(): # Renamed
                x1, y1, x2, y2 = track_data['box'] # Use 'box' (original coords)
                color = (255, 0, 0) if track_data['cmsId'] else (0, 0, 255)
                cv2.rectangle(cam2_display_frame, (x1, y1), (x2, y2), color, 2) # Renamed
                label = f"ID:{face_id} {track_data.get('name', 'Unk')}"
                if track_data.get('status') and track_data.get('name', 'Unk') != 'Unknown': label += f" ({track_data.get('status')})"
                cv2.putText(cam2_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1) # Renamed

            cv2.putText(cam2_display_frame, f"CAM2 IN: {CAM2_PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Renamed
            cv2.putText(cam2_display_frame, f"CAM2 OUT: {CAM2_PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Renamed
        else:
            camera2_stream.stop(); camera2_available = False; cam2_display_frame = None # Renamed

    # --- Display Logic ---
    if not webcam_available and not camera2_available: # Both failed, Renamed
        if initial_webcam_intended or initial_camera2_intended: # If at least one was expected, Renamed
            print("Error: Both cameras are now unavailable. Exiting.")
            break
        else: # No cameras were ever available/intended
             time.sleep(0.1) # Prevent busy loop if no cameras from start
             if cv2.waitKey(1) & 0xFF == ord('q'): break # Allow q to quit
             continue


    display_frame_final = None
    window_title = "Camera Feed"
    display_height, display_width_single = 480, 640
    black_frame = np.zeros((display_height, display_width_single, 3), dtype=np.uint8)
    
    # Use a copy of black_frame if a camera display frame is None
    current_webcam_display_safe = webcam_display_frame if webcam_display_frame is not None else black_frame.copy()
    current_cam2_display_safe = cam2_display_frame if cam2_display_frame is not None else black_frame.copy() # Renamed

    # Ensure frames are of the correct display size (especially black_frame if used)
    if current_webcam_display_safe.shape[0] != display_height or current_webcam_display_safe.shape[1] != display_width_single:
        current_webcam_display_safe = cv2.resize(current_webcam_display_safe, (display_width_single, display_height))
    if current_cam2_display_safe.shape[0] != display_height or current_cam2_display_safe.shape[1] != display_width_single: # Renamed
        current_cam2_display_safe = cv2.resize(current_cam2_display_safe, (display_width_single, display_height)) # Renamed

    if initial_webcam_intended and initial_camera2_intended: # Renamed
        display_frame_final = np.hstack((current_webcam_display_safe, current_cam2_display_safe)) # Renamed
        window_title = 'Camera 0 | Camera 2' # Renamed
    elif initial_webcam_intended:
        display_frame_final = current_webcam_display_safe
        window_title = 'Camera 0 Feed' # Renamed
    elif initial_camera2_intended: # Renamed
        display_frame_final = current_cam2_display_safe # Renamed
        window_title = 'Camera 2 Feed' # Renamed
    
    if display_frame_final is not None and display_frame_final.size > 0 :
        update_latest_frame_flask(display_frame_final) # For Flask stream
        cv2.imshow(window_title, display_frame_final)
    else: # Should only happen if no camera was ever intended
        update_latest_frame_flask(None)
        # If no camera was ever intended, we might not want a window at all,
        # but if one was intended and then failed, we might show "failed"
        if not (initial_webcam_intended or initial_camera2_intended):
            pass # No window if no camera was ever available
        else: # At least one camera was intended but now all are None
            cv2.imshow("Cameras Failed", np.zeros((100,300,3), dtype=np.uint8))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Exiting main loop. Cleaning up...")
if webcam_stream: webcam_stream.stop()
if camera2_stream: camera2_stream.stop() # Renamed
cv2.destroyAllWindows()
# Note: Flask thread is daemon, will exit when main thread exits.
print("Script finished.")
