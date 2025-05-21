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
RECENTLY_LOST_TIMEOUT_SECONDS = 5 # How long to keep a face in the 'recently_lost' cache
MAX_RECOGNITION_ATTEMPTS_PENDING = 3 # Number of re-attempts for a pending face after initial failure

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

# Tracked faces structure:
# { face_id: { 'centroid', 'box', 'last_seen_frame_count',
#              'crossed_A_pending_B', 'crossed_B_pending_A',
#              'name', 'cmsId', 'status', 'live_embedding',
#              'recognized_in_session', 'pending_recognition', 'recognition_attempts_left' } }
tracked_faces = {}
tracked_faces_cam2 = {}
next_face_id = 0
next_face_id_cam2 = 0

recently_lost_faces_webcam = {}
recently_lost_faces_cam2 = {}

MAX_FRAMES_UNSEEN_PROCESSING_CYCLES = 3
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
                                           model_name="Facenet",
                                           enforce_detection=False,
                                           detector_backend='skip',
                                           align=True)
        if embedding_obj and len(embedding_obj) > 0 and "embedding" in embedding_obj[0]:
            return embedding_obj[0]["embedding"]
    except Exception as e:
        # print(f"Error generating live embedding: {e}")
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
    camera_source_for_log = "Block 1"
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
    img_fn = None
    if frame_to_save is not None:
        filename_ts = int(event_timestamp * 1000)
        img_fn = os.path.join(output_dir, f"{base_filename_prefix}_{action_taken}_{filename_ts}.jpg")
        try:
            cv2.imwrite(img_fn, frame_to_save)
            print(f"Saved: {img_fn}")
        except Exception as e:
            print(f"Error saving image {img_fn}: {e}")
            img_fn = None

    log_detection_event(cms_id, person_name, action_taken, camera_source, img_fn, person_status)

# --- Camera Initialization ---
webcam_src = 0
camera2_src = 2

webcam_stream = CameraStream(src=webcam_src, name="Camera0Stream")
camera2_stream = CameraStream(src=camera2_src, name="Camera2Stream")

webcam_available = webcam_stream.isOpened()
camera2_available = camera2_stream.isOpened()

if not webcam_available and not camera2_available:
    print("Error: Neither Camera 0 nor Camera 2 is available. Exiting.")
    if webcam_stream: webcam_stream.stop()
    if camera2_stream: camera2_stream.stop()
    cv2.destroyAllWindows()
    exit()

print("Press 'q' to quit.")
print(f"Recognition threshold (cosine distance): {RECOGNITION_THRESHOLD} (lower is stricter)")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames for AI tasks.")
print(f"Max recognition re-attempts for pending faces: {MAX_RECOGNITION_ATTEMPTS_PENDING}")


if LINE_A_Y >= LINE_B_Y:
    print("Critical Warning: LINE_A_Y should be less than LINE_B_Y for IN=downwards logic.")

webcam_frame_counter = 0
cam2_frame_counter = 0

initial_webcam_intended = webcam_available
initial_camera2_intended = camera2_available

# --- Flask App for Streaming ---
app = Flask(__name__)
latest_frame_flask = None
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
                time.sleep(0.01)
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed_flask():
    return Response(gen_frames_flask(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    print("Starting Flask server on 0.0.0.0:5002")
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# --- Main Loop ---
processing_frame_count_webcam = 0
processing_frame_count_cam2 = 0

while True:
    webcam_display_frame = None
    cam2_display_frame = None
    frame_webcam_original = None
    frame_cam2_original = None

    # --- Process Webcam (Camera 0) ---
    if webcam_available:
        frame_webcam_original = webcam_stream.read()
        if frame_webcam_original is not None:
            webcam_frame_counter += 1
            webcam_display_frame = frame_webcam_original.copy()

            if webcam_frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                processing_frame_count_webcam +=1
                resized_webcam_for_detection = cv2.resize(frame_webcam_original, (640, 480))
                scale_x = frame_webcam_original.shape[1] / 640.0
                scale_y = frame_webcam_original.shape[0] / 480.0

                current_detections_data = []
                results_webcam = model(resized_webcam_for_detection, conf=0.5, verbose=False)
                if results_webcam and results_webcam[0].boxes.data.numel() > 0:
                    for box_data in results_webcam[0].boxes:
                        x1_r, y1_r, x2_r, y2_r = map(int, box_data.xyxy[0])
                        x1_o, y1_o = int(x1_r * scale_x), int(y1_r * scale_y)
                        x2_o, y2_o = int(x2_r * scale_x), int(y2_r * scale_y)

                        face_roi_original = None
                        if y1_o < y2_o and x1_o < x2_o:
                             face_roi_original = frame_webcam_original[max(0,y1_o):min(frame_webcam_original.shape[0],y2_o),
                                                                       max(0,x1_o):min(frame_webcam_original.shape[1],x2_o)]

                        cx_o = (x1_o + x2_o) // 2
                        cy_o = (y1_o + y2_o) // 2
                        current_detections_data.append({
                            'centroid': (cx_o, cy_o),
                            'box': (x1_o, y1_o, x2_o, y2_o),
                            'face_roi_original': face_roi_original
                        })

                matched_track_ids_this_cycle = set()
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
                            'last_seen_frame_count': processing_frame_count_webcam
                        })
                        action_taken = None
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

                        if track.get('pending_recognition', False) and track.get('recognition_attempts_left', 0) > 0:
                            current_roi_for_rerecog = det_data.get('face_roi_original')
                            if current_roi_for_rerecog is not None and current_roi_for_rerecog.size > 0:
                                live_emb_rerecog = create_face_embedding_live(current_roi_for_rerecog)
                                if live_emb_rerecog:
                                    track['live_embedding'] = live_emb_rerecog # Update with latest embedding
                                    name_r, cmsId_r, status_r, dist_r = find_match_against_known_db(live_emb_rerecog, known_face_embeddings, RECOGNITION_THRESHOLD)
                                    if cmsId_r:
                                        print(f"Webcam: Pending track ID {best_match_id} RECOGNIZED as {name_r} (CMS: {cmsId_r}). Dist: {dist_r:.3f}")
                                        track.update({
                                            'name': name_r, 'cmsId': cmsId_r, 'status': status_r,
                                            'pending_recognition': False,
                                            'recognition_attempts_left': 0,
                                            'recognized_in_session': True
                                        })
                                    else:
                                        track['recognition_attempts_left'] -= 1
                                        if track['recognition_attempts_left'] <= 0:
                                            track['pending_recognition'] = False
                                            track['name'] = "Unknown" # Finalize as Unknown
                                            print(f"Webcam: Pending track ID {best_match_id} finalized as Unknown after max re-attempts.")
                                else:
                                    track['recognition_attempts_left'] -= 1
                                    if track['recognition_attempts_left'] <= 0:
                                        track['pending_recognition'] = False; track['name'] = "Unknown"
                            else:
                                track['recognition_attempts_left'] -= 1
                                if track['recognition_attempts_left'] <= 0:
                                    track['pending_recognition'] = False; track['name'] = "Unknown"
                        current_detections_data[det_idx] = None

                current_detections_data = [d for d in current_detections_data if d is not None]

                lost_ids = []
                for face_id, track_data in tracked_faces.items():
                    if face_id not in matched_track_ids_this_cycle:
                        if (processing_frame_count_webcam - track_data['last_seen_frame_count']) > MAX_FRAMES_UNSEEN_PROCESSING_CYCLES:
                            lost_ids.append(face_id)
                            if not track_data.get('pending_recognition', False) and track_data.get('live_embedding'):
                                print(f"Webcam: Moving finalized track ID {face_id} ({track_data['name']}) to lost cache.")
                                recently_lost_faces_webcam[face_id] = {
                                    'name': track_data['name'], 'cmsId': track_data['cmsId'],
                                    'status': track_data['status'], 'last_embedding': track_data['live_embedding'],
                                    'lost_timestamp': time.time(), 'last_box': track_data['box']
                                }
                for face_id in lost_ids: del tracked_faces[face_id]

                current_time = time.time()
                for lost_id, lost_data in list(recently_lost_faces_webcam.items()):
                    if current_time - lost_data['lost_timestamp'] > RECENTLY_LOST_TIMEOUT_SECONDS:
                        del recently_lost_faces_webcam[lost_id]

                for new_det_data in current_detections_data:
                    live_emb = create_face_embedding_live(new_det_data['face_roi_original'])

                    initial_name, initial_cmsId, initial_status, initial_dist = "Unknown", None, None, float('inf')
                    is_pending = True
                    attempts_left = MAX_RECOGNITION_ATTEMPTS_PENDING
                    is_recognized_session = False
                    current_track_name_on_creation = "Pending"

                    revived_id = None
                    if live_emb:
                        best_lost_match_val = float('inf')
                        for lost_id, lost_data_entry in recently_lost_faces_webcam.items():
                            if live_emb and lost_data_entry.get('last_embedding'): # Check if both embeddings are valid
                                try:
                                    dist_to_lost = cosine(np.array(live_emb).flatten(), np.array(lost_data_entry['last_embedding']).flatten())
                                    if dist_to_lost < RECOGNITION_RE_MATCH_THRESHOLD and dist_to_lost < best_lost_match_val:
                                        best_lost_match_val = dist_to_lost
                                        revived_id = lost_id
                                except ValueError as ve: # Handle potential shape mismatches or empty arrays
                                    # print(f"Webcam: Cosine distance error for lost cache check: {ve}")
                                    pass # Skip this comparison

                        if revived_id:
                            matched_lost_data = recently_lost_faces_webcam[revived_id]
                            
                            # ---- START OF UPDATED LOGIC FOR CACHE HIT ----
                            if matched_lost_data.get('cmsId') is None: # If cached was "Unknown"
                                print(f"Webcam: Re-ID from lost cache was '{matched_lost_data.get('name', 'Unknown')}' (ID {revived_id}). Attempting DB lookup for potential upgrade. Cache dist: {best_lost_match_val:.3f}")
                                name_db, cmsId_db, status_db, dist_db = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                                if cmsId_db: # Found a better match in the full DB
                                    print(f"Webcam: Upgraded cached 'Unknown' to '{name_db}' (CMS: {cmsId_db}) from DB. Dist: {dist_db:.3f}")
                                    initial_name, initial_cmsId, initial_status, initial_dist = name_db, cmsId_db, status_db, dist_db
                                    current_track_name_on_creation = initial_name
                                    is_pending = False
                                    attempts_left = 0
                                    is_recognized_session = True # Recognized from DB
                                else: # Still unknown after DB check, stick with cached "Unknown"
                                    print(f"Webcam: DB lookup for cached 'Unknown' also resulted in Unknown. Keeping cached info.")
                                    initial_name, initial_cmsId, initial_status, initial_dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val
                                    current_track_name_on_creation = initial_name # Will be "Unknown"
                                    is_pending = False # Was already finalized as Unknown when lost
                                    attempts_left = 0
                                    is_recognized_session = True # Re-ID'd from cache, even if Unknown
                            else: # Cached entry was already a known person
                                print(f"Webcam: Re-ID {matched_lost_data['name']} (CMS: {matched_lost_data['cmsId']}) from lost cache. Dist: {best_lost_match_val:.3f}")
                                initial_name, initial_cmsId, initial_status, initial_dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val
                                current_track_name_on_creation = initial_name
                                is_pending = False
                                attempts_left = 0
                                is_recognized_session = True # Re-ID'd from cache
                            # ---- END OF UPDATED LOGIC FOR CACHE HIT ----
                            del recently_lost_faces_webcam[revived_id]
                        else: # Not in lost cache, try full DB for the *first time* for this detection
                            initial_name, initial_cmsId, initial_status, initial_dist = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                            if initial_cmsId: # Initial recognition successful
                                current_track_name_on_creation = initial_name
                                is_pending = False
                                attempts_left = 0
                                is_recognized_session = True
                            # else: it remains pending (is_pending=True by default), current_track_name_on_creation is "Pending"

                    new_id = next_face_id
                    next_face_id += 1
                    tracked_faces[new_id] = {
                        'centroid': new_det_data['centroid'], 'box': new_det_data['box'],
                        'last_seen_frame_count': processing_frame_count_webcam,
                        'crossed_A_pending_B': False, 'crossed_B_pending_A': False,
                        'name': current_track_name_on_creation,
                        'cmsId': initial_cmsId, 'status': initial_status,
                        'live_embedding': live_emb, # Store first embedding
                        'recognized_in_session': is_recognized_session,
                        'pending_recognition': is_pending,
                        'recognition_attempts_left': attempts_left
                    }
                    if is_pending:
                         print(f"Webcam: New track ID {new_id} is PENDING recognition. Initial attempt dist: {(f'{initial_dist:.3f}' if initial_dist != float('inf') else 'inf') if live_emb else 'N/A'}")
                    elif not revived_id and initial_cmsId: # Recognized on first try, not from cache, and actually recognized
                         print(f"Webcam: New track ID {new_id} ({current_track_name_on_creation}) recognized. Dist: {initial_dist:.3f}")
                    # No specific print if it's not pending, not revived, and not recognized (i.e., becomes "Unknown" immediately but is_pending=False)
                    # This case is less common with the current logic flow but handled by the `current_track_name_on_creation` default if live_emb is None.


            for face_id, track_data in tracked_faces.items():
                x1, y1, x2, y2 = track_data['box']
                color = (255, 165, 0) if track_data.get('pending_recognition', False) else \
                        ((255, 0, 0) if track_data.get('cmsId') else (0, 0, 255)) # Orange for pending
                cv2.rectangle(webcam_display_frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{face_id} {track_data.get('name', 'Unk')}"
                if track_data.get('pending_recognition', False):
                    attempts_made = MAX_RECOGNITION_ATTEMPTS_PENDING - track_data.get('recognition_attempts_left', 0) +1
                    label += f" ({attempts_made}/{MAX_RECOGNITION_ATTEMPTS_PENDING})"
                elif track_data.get('status') and track_data.get('name', 'Unk') != 'Unknown' and track_data.get('cmsId'):
                     label += f" ({track_data.get('status')})"
                cv2.putText(webcam_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(webcam_display_frame, f"IN: {PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(webcam_display_frame, f"OUT: {PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            webcam_stream.stop(); webcam_available = False; webcam_display_frame = None

    # --- Process Camera 2 (formerly IP camera) ---
    if camera2_available:
        frame_cam2_original = camera2_stream.read()
        if frame_cam2_original is not None:
            cam2_frame_counter += 1
            cam2_display_frame = frame_cam2_original.copy()

            if cam2_frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                processing_frame_count_cam2 +=1

                resized_cam2_for_detection = cv2.resize(frame_cam2_original, (640, 480))
                scale_x_cam2 = frame_cam2_original.shape[1] / 640.0
                scale_y_cam2 = frame_cam2_original.shape[0] / 480.0

                current_detections_data_cam2 = []
                results_cam2 = model(resized_cam2_for_detection, conf=0.5, verbose=False)
                if results_cam2 and results_cam2[0].boxes.data.numel() > 0:
                    for box_data in results_cam2[0].boxes:
                        if int(box_data.cls[0].item()) == FACE_CLASS_ID:
                            x1_r, y1_r, x2_r, y2_r = map(int, box_data.xyxy[0])
                            x1_o, y1_o = int(x1_r * scale_x_cam2), int(y1_r * scale_y_cam2)
                            x2_o, y2_o = int(x2_r * scale_x_cam2), int(y2_r * scale_y_cam2)

                            face_roi_original_cam2 = None
                            if y1_o < y2_o and x1_o < x2_o:
                                face_roi_original_cam2 = frame_cam2_original[max(0,y1_o):min(frame_cam2_original.shape[0],y2_o),
                                                                             max(0,x1_o):min(frame_cam2_original.shape[1],x2_o)]

                            cx_o = (x1_o + x2_o) // 2
                            cy_o = (y1_o + y2_o) // 2
                            current_detections_data_cam2.append({
                                'centroid': (cx_o, cy_o),
                                'box': (x1_o, y1_o, x2_o, y2_o),
                                'face_roi_original': face_roi_original_cam2
                            })

                matched_track_ids_cam2_this_cycle = set()
                for det_idx, det_data in enumerate(current_detections_data_cam2):
                    best_match_id = -1
                    min_dist_centroid = CENTROID_MATCH_THRESHOLD
                    for face_id, track_data in tracked_faces_cam2.items():
                        dist = math.dist(track_data['centroid'], det_data['centroid'])
                        if dist < min_dist_centroid:
                            min_dist_centroid = dist
                            best_match_id = face_id

                    if best_match_id != -1:
                        matched_track_ids_cam2_this_cycle.add(best_match_id)
                        track = tracked_faces_cam2[best_match_id]
                        prev_cy = track['centroid'][1]
                        curr_cy = det_data['centroid'][1]
                        track.update({
                            'centroid': det_data['centroid'],
                            'box': det_data['box'],
                            'last_seen_frame_count': processing_frame_count_cam2
                        })
                        action_taken = None
                        if not track['crossed_A_pending_B'] and prev_cy < LINE_A_Y and curr_cy >= LINE_A_Y: track['crossed_A_pending_B'] = True; track['crossed_B_pending_A'] = False
                        elif not track['crossed_B_pending_A'] and prev_cy >= LINE_B_Y and curr_cy < LINE_B_Y: track['crossed_B_pending_A'] = True; track['crossed_A_pending_B'] = False
                        if track['crossed_A_pending_B'] and prev_cy < LINE_B_Y and curr_cy >= LINE_B_Y: CAM2_PERSONS_OUT_COUNT += 1; action_taken = "OUT"; track['crossed_A_pending_B'] = False
                        elif track['crossed_B_pending_A'] and prev_cy >= LINE_A_Y and curr_cy < LINE_A_Y: CAM2_PERSONS_IN_COUNT += 1; action_taken = "IN"; track['crossed_B_pending_A'] = False
                        if track['crossed_A_pending_B'] and curr_cy < LINE_A_Y: track['crossed_A_pending_B'] = False
                        if track['crossed_B_pending_A'] and curr_cy >= LINE_B_Y: track['crossed_B_pending_A'] = False

                        if action_taken:
                            event_time_cam2 = time.time()
                            print(f"[CAM2] ID {best_match_id} ({track.get('name', 'Unknown')}) COUNTED {action_taken}. Total IN: {CAM2_PERSONS_IN_COUNT}, Total OUT: {CAM2_PERSONS_OUT_COUNT}")
                            frame_copy_for_saving_cam2 = frame_cam2_original.copy() if frame_cam2_original is not None else None
                            filename_prefix_cam2 = f"{('recognized_cam2_' + str(track['cmsId'])) if track.get('cmsId') else 'unknown_camera2'}"
                            threading.Thread(target=save_and_log_event_async, args=(
                                frame_copy_for_saving_cam2, OUTPUT_DIR, filename_prefix_cam2, action_taken,
                                track.get('cmsId'), track.get('name'), "camera2", track.get('status'), event_time_cam2
                            )).start()

                        if track.get('pending_recognition', False) and track.get('recognition_attempts_left', 0) > 0:
                            current_roi_for_rerecog = det_data.get('face_roi_original')
                            if current_roi_for_rerecog is not None and current_roi_for_rerecog.size > 0:
                                live_emb_rerecog = create_face_embedding_live(current_roi_for_rerecog)
                                if live_emb_rerecog:
                                    track['live_embedding'] = live_emb_rerecog # Update with latest embedding
                                    name_r, cmsId_r, status_r, dist_r = find_match_against_known_db(live_emb_rerecog, known_face_embeddings, RECOGNITION_THRESHOLD)
                                    if cmsId_r:
                                        print(f"Camera2: Pending track ID {best_match_id} RECOGNIZED as {name_r} (CMS: {cmsId_r}). Dist: {dist_r:.3f}")
                                        track.update({
                                            'name': name_r, 'cmsId': cmsId_r, 'status': status_r,
                                            'pending_recognition': False,
                                            'recognition_attempts_left': 0,
                                            'recognized_in_session': True
                                        })
                                    else:
                                        track['recognition_attempts_left'] -= 1
                                        if track['recognition_attempts_left'] <= 0:
                                            track['pending_recognition'] = False
                                            track['name'] = "Unknown" # Finalize as Unknown
                                            print(f"Camera2: Pending track ID {best_match_id} finalized as Unknown after max re-attempts.")
                                else:
                                    track['recognition_attempts_left'] -= 1
                                    if track['recognition_attempts_left'] <= 0:
                                        track['pending_recognition'] = False; track['name'] = "Unknown"
                            else:
                                track['recognition_attempts_left'] -= 1
                                if track['recognition_attempts_left'] <= 0:
                                     track['pending_recognition'] = False; track['name'] = "Unknown"
                        current_detections_data_cam2[det_idx] = None

                current_detections_data_cam2 = [d for d in current_detections_data_cam2 if d is not None]

                lost_ids_cam2 = []
                for face_id, track_data in tracked_faces_cam2.items():
                    if face_id not in matched_track_ids_cam2_this_cycle:
                        if (processing_frame_count_cam2 - track_data['last_seen_frame_count']) > MAX_FRAMES_UNSEEN_PROCESSING_CYCLES:
                            lost_ids_cam2.append(face_id)
                            if not track_data.get('pending_recognition', False) and track_data.get('live_embedding'):
                                print(f"Camera2: Moving finalized track ID {face_id} ({track_data['name']}) to lost cache.")
                                recently_lost_faces_cam2[face_id] = {
                                    'name': track_data['name'], 'cmsId': track_data['cmsId'],
                                    'status': track_data['status'], 'last_embedding': track_data['live_embedding'],
                                    'lost_timestamp': time.time(), 'last_box': track_data['box']
                                }
                for face_id in lost_ids_cam2: del tracked_faces_cam2[face_id]

                current_time_cam2 = time.time()
                for lost_id, lost_data in list(recently_lost_faces_cam2.items()):
                    if current_time_cam2 - lost_data['lost_timestamp'] > RECENTLY_LOST_TIMEOUT_SECONDS:
                        del recently_lost_faces_cam2[lost_id]

                for new_det_data in current_detections_data_cam2:
                    live_emb = create_face_embedding_live(new_det_data['face_roi_original'])

                    initial_name, initial_cmsId, initial_status, initial_dist = "Unknown", None, None, float('inf')
                    is_pending = True
                    attempts_left = MAX_RECOGNITION_ATTEMPTS_PENDING
                    is_recognized_session = False
                    current_track_name_on_creation = "Pending"

                    revived_id_cam2 = None
                    if live_emb:
                        best_lost_match_val_cam2 = float('inf')
                        for lost_id, lost_data_entry in recently_lost_faces_cam2.items():
                            if live_emb and lost_data_entry.get('last_embedding'): # Check if both embeddings are valid
                                try:
                                    dist_to_lost = cosine(np.array(live_emb).flatten(), np.array(lost_data_entry['last_embedding']).flatten())
                                    if dist_to_lost < RECOGNITION_RE_MATCH_THRESHOLD and dist_to_lost < best_lost_match_val_cam2:
                                        best_lost_match_val_cam2 = dist_to_lost
                                        revived_id_cam2 = lost_id
                                except ValueError as ve: # Handle potential shape mismatches or empty arrays
                                    # print(f"Camera2: Cosine distance error for lost cache check: {ve}")
                                    pass # Skip this comparison

                        if revived_id_cam2:
                            matched_lost_data = recently_lost_faces_cam2[revived_id_cam2]
                            
                            # ---- START OF UPDATED LOGIC FOR CACHE HIT (CAM2) ----
                            if matched_lost_data.get('cmsId') is None: # If cached was "Unknown"
                                print(f"Camera2: Re-ID from lost cache was '{matched_lost_data.get('name', 'Unknown')}' (ID {revived_id_cam2}). Attempting DB lookup for potential upgrade. Cache dist: {best_lost_match_val_cam2:.3f}")
                                name_db, cmsId_db, status_db, dist_db = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                                if cmsId_db: # Found a better match in the full DB
                                    print(f"Camera2: Upgraded cached 'Unknown' to '{name_db}' (CMS: {cmsId_db}) from DB. Dist: {dist_db:.3f}")
                                    initial_name, initial_cmsId, initial_status, initial_dist = name_db, cmsId_db, status_db, dist_db
                                    current_track_name_on_creation = initial_name
                                    is_pending = False
                                    attempts_left = 0
                                    is_recognized_session = True # Recognized from DB
                                else: # Still unknown after DB check, stick with cached "Unknown"
                                    print(f"Camera2: DB lookup for cached 'Unknown' also resulted in Unknown. Keeping cached info.")
                                    initial_name, initial_cmsId, initial_status, initial_dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val_cam2
                                    current_track_name_on_creation = initial_name # Will be "Unknown"
                                    is_pending = False # Was already finalized as Unknown when lost
                                    attempts_left = 0
                                    is_recognized_session = True # Re-ID'd from cache, even if Unknown
                            else: # Cached entry was already a known person
                                print(f"Camera2: Re-ID {matched_lost_data['name']} (CMS: {matched_lost_data['cmsId']}) from lost cache. Dist: {best_lost_match_val_cam2:.3f}")
                                initial_name, initial_cmsId, initial_status, initial_dist = matched_lost_data['name'], matched_lost_data['cmsId'], matched_lost_data['status'], best_lost_match_val_cam2
                                current_track_name_on_creation = initial_name
                                is_pending = False
                                attempts_left = 0
                                is_recognized_session = True # Re-ID'd from cache
                            # ---- END OF UPDATED LOGIC FOR CACHE HIT (CAM2) ----
                            del recently_lost_faces_cam2[revived_id_cam2]
                        else: # Not in lost cache, try full DB
                            initial_name, initial_cmsId, initial_status, initial_dist = find_match_against_known_db(live_emb, known_face_embeddings, RECOGNITION_THRESHOLD)
                            if initial_cmsId:
                                current_track_name_on_creation = initial_name
                                is_pending = False
                                attempts_left = 0
                                is_recognized_session = True
                    
                    new_id = next_face_id_cam2
                    next_face_id_cam2 += 1
                    tracked_faces_cam2[new_id] = {
                        'centroid': new_det_data['centroid'], 'box': new_det_data['box'],
                        'last_seen_frame_count': processing_frame_count_cam2,
                        'crossed_A_pending_B': False, 'crossed_B_pending_A': False,
                        'name': current_track_name_on_creation,
                        'cmsId': initial_cmsId, 'status': initial_status,
                        'live_embedding': live_emb,
                        'recognized_in_session': is_recognized_session,
                        'pending_recognition': is_pending,
                        'recognition_attempts_left': attempts_left
                    }
                    if is_pending:
                        print(f"Camera2: New track ID {new_id} is PENDING recognition. Initial attempt dist: {(f'{initial_dist:.3f}' if initial_dist != float('inf') else 'inf') if live_emb else 'N/A'}")
                    elif not revived_id_cam2 and initial_cmsId: # Recognized on first try, not from cache, and actually recognized
                        print(f"Camera2: New track ID {new_id} ({current_track_name_on_creation}) recognized. Dist: {initial_dist:.3f}")


            for face_id, track_data in tracked_faces_cam2.items():
                x1, y1, x2, y2 = track_data['box']
                color = (255, 165, 0) if track_data.get('pending_recognition', False) else \
                        ((255, 0, 0) if track_data.get('cmsId') else (0, 0, 255))
                cv2.rectangle(cam2_display_frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{face_id} {track_data.get('name', 'Unk')}"
                if track_data.get('pending_recognition', False):
                    attempts_made = MAX_RECOGNITION_ATTEMPTS_PENDING - track_data.get('recognition_attempts_left', 0) + 1
                    label += f" ({attempts_made}/{MAX_RECOGNITION_ATTEMPTS_PENDING})"
                elif track_data.get('status') and track_data.get('name', 'Unk') != 'Unknown' and track_data.get('cmsId'):
                     label += f" ({track_data.get('status')})"
                cv2.putText(cam2_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(cam2_display_frame, f"CAM2 IN: {CAM2_PERSONS_IN_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(cam2_display_frame, f"CAM2 OUT: {CAM2_PERSONS_OUT_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            camera2_stream.stop(); camera2_available = False; cam2_display_frame = None

    # --- Display Logic ---
    if not webcam_available and not camera2_available:
        if initial_webcam_intended or initial_camera2_intended:
            print("Error: Both cameras are now unavailable. Exiting.")
            break
        else:
             time.sleep(0.1)
             if cv2.waitKey(1) & 0xFF == ord('q'): break
             continue


    display_frame_final = None
    window_title = "Camera Feed"
    display_height, display_width_single = 480, 640
    black_frame = np.zeros((display_height, display_width_single, 3), dtype=np.uint8)

    current_webcam_display_safe = webcam_display_frame if webcam_display_frame is not None else black_frame.copy()
    current_cam2_display_safe = cam2_display_frame if cam2_display_frame is not None else black_frame.copy()

    if current_webcam_display_safe.shape[0] != display_height or current_webcam_display_safe.shape[1] != display_width_single:
        current_webcam_display_safe = cv2.resize(current_webcam_display_safe, (display_width_single, display_height))
    if current_cam2_display_safe.shape[0] != display_height or current_cam2_display_safe.shape[1] != display_width_single:
        current_cam2_display_safe = cv2.resize(current_cam2_display_safe, (display_width_single, display_height))

    if initial_webcam_intended and initial_camera2_intended:
        display_frame_final = np.hstack((current_webcam_display_safe, current_cam2_display_safe))
        window_title = 'Camera 0 | Camera 2'
    elif initial_webcam_intended:
        display_frame_final = current_webcam_display_safe
        window_title = 'Camera 0 Feed'
    elif initial_camera2_intended:
        display_frame_final = current_cam2_display_safe
        window_title = 'Camera 2 Feed'

    if display_frame_final is not None and display_frame_final.size > 0 :
        update_latest_frame_flask(display_frame_final)
        cv2.imshow(window_title, display_frame_final)
    else:
        update_latest_frame_flask(None)
        if not (initial_webcam_intended or initial_camera2_intended):
            pass
        else:
            cv2.imshow("Cameras Failed", np.zeros((100,300,3), dtype=np.uint8))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Exiting main loop. Cleaning up...")
if webcam_stream: webcam_stream.stop()
if camera2_stream: camera2_stream.stop()
cv2.destroyAllWindows()
print("Script finished.")
