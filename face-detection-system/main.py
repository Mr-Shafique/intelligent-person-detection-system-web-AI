import cv2
import os
import time
import numpy as np
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import pickle
import requests
import traceback
import base64  # Add this import for encoding images

# --- Configuration ---
DEEPFACE_MODEL = "VGG-Face"
DEEPFACE_METRIC = "cosine"
MATCH_THRESHOLD = 0.65  # Threshold for local duplicate check
ONLINE_MATCH_THRESHOLD = 0.65 # Threshold for matching against online DB (can be adjusted)

# Local Capture Config
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_faces")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_embeddings.pkl") # Local captures
CAPTURE_INTERVAL = 0.333  # 1 image every 0.333 seconds = 3 images per second
FACE_PADDING = 30

# Online API Config
ONLINE_EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_face_embeddings.pkl") # Online data
API_ENDPOINT = "http://localhost:5000/api/persons"
BACKEND_URL = "http://localhost:5000" # Base URL for images
DETECTIONS_API_ENDPOINT = "http://localhost:5000/api/detections"  # Add this for detection log endpoint

# Storage for face embeddings
face_database = []  # Local captures: {path: str, embedding: np.array}
online_face_database = [] # Online data: {name: str, cmsId: str, status: str, _id: str, embedding: np.array}

# Create output directory
os.makedirs(CAPTURE_DIR, exist_ok=True)
print(f"Images will be saved to: {os.path.abspath(CAPTURE_DIR)}")
print(f"Embeddings database file: {os.path.abspath(EMBEDDINGS_FILE)}")
print(f"Online embeddings database file: {os.path.abspath(ONLINE_EMBEDDINGS_FILE)}")

# Function to save the LOCAL database to a file
def save_database():
    try:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(face_database, f)
    except Exception as e:
        print(f"Error saving database to {EMBEDDINGS_FILE}: {e}")

# Function to save the ONLINE database to a file
def save_online_database():
    print(f"[SAVE_ONLINE] Attempting to save {len(online_face_database)} entries to {ONLINE_EMBEDDINGS_FILE}")
    try:
        with open(ONLINE_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(online_face_database, f)
        print(f"[SAVE_ONLINE] Successfully saved {len(online_face_database)} entries.")
    except Exception as e:
        print(f"[SAVE_ONLINE] Error saving online database to {ONLINE_EMBEDDINGS_FILE}: {e}")
        traceback.print_exc()

# Function to load existing LOCAL face embeddings
def load_existing_faces():
    global face_database
    database_updated = False
    known_paths = set()
    file_existed_initially = os.path.exists(EMBEDDINGS_FILE)

    if file_existed_initially:
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                face_database = pickle.load(f)
            print(f"Loaded {len(face_database)} embeddings from {EMBEDDINGS_FILE}")
            for entry in face_database:
                known_paths.add(entry.get('path'))
        except Exception as e:
            print(f"Error loading database from {EMBEDDINGS_FILE}: {e}. Initializing empty database.")
            face_database = []
            file_existed_initially = False
    else:
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Initializing empty database.")
        face_database = []

    print(f"Scanning {CAPTURE_DIR} for new faces...")
    new_faces_found = 0
    for filename in os.listdir(CAPTURE_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        filepath = os.path.join(CAPTURE_DIR, filename)

        if filepath in known_paths:
            continue

        print(f"Processing new image: {filename}")
        new_faces_found += 1
        try:
            embedding_objs = DeepFace.represent(
                img_path=filepath,
                model_name=DEEPFACE_MODEL,
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )

            if embedding_objs:
                embedding = embedding_objs[0]['embedding']
                face_database.append({
                    'path': filepath,
                    'embedding': embedding
                })
                known_paths.add(filepath)
                database_updated = True
                if new_faces_found % 10 == 0:
                     print(f"Added {new_faces_found} new faces to database...")
            else:
                 print(f"Warning: No embedding generated for new file {filename}")

        except Exception as e:
            print(f"Error processing new image {filepath}: {e}")

    print(f"Found and processed {new_faces_found} new faces from directory scan.")

    if database_updated:
        print("Database was updated by scan, saving changes...")
        save_database()
    elif not file_existed_initially:
        print(f"Embeddings file did not exist, creating it now (even if empty)...")
        save_database()

    print(f"Database ready with {len(face_database)} total entries.")


# Function to load ONLINE face embeddings from API and file (Modified to always refresh from API)
def load_online_faces():
    global online_face_database
    online_face_database = []
    print(f"[LOAD_ONLINE] Initialized empty online face database. Current size: {len(online_face_database)}")

    print(f"[LOAD_ONLINE] Fetching person data from API: {API_ENDPOINT}")
    persons_data = []
    try:
        response = requests.get(API_ENDPOINT, timeout=20)
        print(f"[LOAD_ONLINE] API response status code: {response.status_code}")
        response.raise_for_status()
        persons_data = response.json()
        print(f"[LOAD_ONLINE] Received data for {len(persons_data)} persons from API.")

        if not isinstance(persons_data, list):
             print(f"[LOAD_ONLINE] Error: API response is not a list. Received type: {type(persons_data)}")
             persons_data = []

    except requests.exceptions.RequestException as e:
        print(f"[LOAD_ONLINE] FATAL: Error fetching data from API ({API_ENDPOINT}): {e}")
    except Exception as e:
        print(f"[LOAD_ONLINE] FATAL: Error processing API response: {e}")
        traceback.print_exc()

    api_faces_processed_count = 0
    image_fields = ['frontImage', 'leftImage', 'rightImage']

    print(f"[LOAD_ONLINE] Starting processing for {len(persons_data)} persons.")
    for i, person in enumerate(persons_data):
        cmsId = person.get('cmsId')
        name = person.get('name')
        status = person.get('status', 'unknown')
        person_mongo_id = person.get('_id')
        print(f"[LOAD_ONLINE] Person {i+1}/{len(persons_data)}: ID={cmsId}, Name={name}, MongoDB_ID={person_mongo_id}")
        if not cmsId or not name or not person_mongo_id:
            print(f"  [LOAD_ONLINE] Warning: Skipping person due to missing cmsId, name, or _id: {person}")
            continue

        person_embeddings_added = 0
        valid_image_found = False
        for field in image_fields:
            relative_image_path = person.get(field)
            if not relative_image_path:
                continue

            valid_image_found = True
            image_url = f"{BACKEND_URL}{relative_image_path}"

            try:
                img_response = requests.get(image_url, timeout=10)
                img_response.raise_for_status()
                image_data = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"    [LOAD_ONLINE] Warning: Could not decode image from {image_url}")
                    continue

                embedding_objs = DeepFace.represent(
                    img_path=img,
                    model_name=DEEPFACE_MODEL,
                    enforce_detection=True,
                    detector_backend='opencv',
                    align=True
                )

                if embedding_objs:
                    embedding = embedding_objs[0]['embedding']
                    online_face_database.append({
                        'name': name, 'cmsId': cmsId, 'status': status, '_id': person_mongo_id,
                        'embedding': embedding, 'source_image': relative_image_path
                    })
                    api_faces_processed_count += 1
                    person_embeddings_added += 1
                    print(f"    [LOAD_ONLINE] Success: Added embedding from {field}.")
                else:
                    print(f"    [LOAD_ONLINE] Warning: No face detected by DeepFace in image from {field} ({image_url})")

            except requests.exceptions.RequestException as img_req_err:
                print(f"    [LOAD_ONLINE] Error fetching image {image_url}: {img_req_err}")
            except Exception as img_proc_err:
                print(f"    [LOAD_ONLINE] Error processing image {image_url}: {img_proc_err}")

        if person_embeddings_added == 0:
             if not valid_image_found:
                 print(f"  [LOAD_ONLINE] -> No embeddings generated for {name} ({cmsId}) because no valid image URLs were found in API data.")
             else:
                 print(f"  [LOAD_ONLINE] -> No embeddings generated for {name} ({cmsId}) despite trying images. Check warnings above (decode/detection failures).")

    print(f"[LOAD_ONLINE] Finished processing API data. Generated {api_faces_processed_count} embeddings this run.")
    print(f"[LOAD_ONLINE] Final in-memory online_face_database size before save: {len(online_face_database)}")

    save_online_database()

    print(f"[LOAD_ONLINE] Online database refresh process complete. Final in-memory list size: {len(online_face_database)}")


# Function to find a match in the ONLINE database
def find_online_match(face_embedding):
    if not online_face_database:
        return None

    min_distance = float('inf')
    best_match = None

    for entry in online_face_database:
        stored_embedding = entry.get('embedding')
        if stored_embedding is None:
            continue

        try:
            distance = cosine(face_embedding, stored_embedding)
            if distance < min_distance:
                min_distance = distance
                if distance < ONLINE_MATCH_THRESHOLD:
                    best_match = {
                        'name': entry.get('name', 'N/A'),
                        'status': entry.get('status', 'N/A'),
                        '_id': entry.get('_id'),
                        'cmsId': entry.get('cmsId', 'N/A'),
                        'distance': distance
                    }
        except Exception as e:
            print(f"[Online Match] Error calculating distance: {e}")
            continue

    if best_match:
        print(f"[Debug] Online Match Found: {best_match['name']} ({best_match['status']}) - Dist: {best_match['distance']:.4f}")
        return best_match
    else:
        return None


# Function to check if a face already exists in our LOCAL database (accepts embedding)
def is_local_duplicate(face_embedding):
    min_distance_found = float('inf')
    matched_file = None

    if not face_database:
         return False

    try:
        for stored_face in face_database:
            stored_embedding = stored_face['embedding']
            distance = cosine(face_embedding, stored_embedding)

            if distance < min_distance_found:
                min_distance_found = distance
                matched_file = os.path.basename(stored_face['path'])

            if distance < MATCH_THRESHOLD:
                print(f"[Debug] Local Duplicate Check: Match found! vs {os.path.basename(stored_face['path'])} -> Distance: {distance:.4f}")
                return True

        return False

    except Exception as e:
        print(f"Error checking for local duplicate: {e}")
        return False


# Camera Configuration
WEBCAM_INDEX = 0
IP_CAMERA_URLS = [
    "http://10.102.164.70:8080/video",
    "rtsp://10.102.164.70:554/live",
    "https://10.102.164.70:8080"
]

# Function to attempt IP camera connection with multiple URLs
def try_connect_ip_camera():
    print("Attempting to connect to IP Camera using multiple URL formats...")
    
    for url in IP_CAMERA_URLS:
        print(f"Trying: {url}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"✓ Successfully connected to IP camera using: {url}")
                return cap, url
            else:
                print(f"× Connection opened but couldn't read frame from: {url}")
                cap.release()
        else:
            print(f"× Failed to connect using: {url}")
    
    print("× All connection attempts to IP camera failed.")
    return None, None

# Initialize YOLO face detector
print("Loading YOLOv8 face detection model...")
model = YOLO('yolov8n-face.pt')
print("YOLO model loaded.")

# Initialize cameras
print(f"Initializing Webcam (Index: {WEBCAM_INDEX})...")
cap_webcam = cv2.VideoCapture(WEBCAM_INDEX)
webcam_available = cap_webcam.isOpened()

if webcam_available:
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam initialized.")
else:
    print("Warning: Webcam not available!")

print("Initializing IP Camera...")
cap_ipcam, connected_url = try_connect_ip_camera()
ipcam_available = cap_ipcam is not None and cap_ipcam.isOpened()

if ipcam_available:
    print(f"IP Camera initialized using URL: {connected_url}")
else:
    print("Warning: IP Camera not available!")

if not webcam_available and not ipcam_available:
    print("Error: No cameras are available. Exiting...")
    exit(1)

load_existing_faces()
load_online_faces()

camera_states = {
    'webcam': {'last_capture_time': 0, 'faces_captured': 0, 'duplicates_skipped': 0},
    'ipcam': {'last_capture_time': 0, 'faces_captured': 0, 'duplicates_skipped': 0}
}

print("Starting face detection. Press 'q' to quit.")

def process_frame(frame, camera_id, camera_state):
    if frame is None:
        print(f"[{camera_id}] Error: Received None frame for processing.")
        return np.zeros((480, 640, 3), dtype=np.uint8), camera_state

    display_frame = frame.copy()
    current_time = time.time()
    processed_in_interval = False

    try:
        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 and results[0].boxes is not None else []
    except Exception as e:
        print(f"[{camera_id}] Error during YOLO detection: {e}")
        boxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        w = x2 - x1
        h = y2 - y1

        if w < 20 or h < 20:
            continue

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not processed_in_interval and (current_time - camera_state['last_capture_time'] >= CAPTURE_INTERVAL):
            processed_in_interval = True
            camera_state['last_capture_time'] = current_time

            frame_h, frame_w = frame.shape[:2]
            pad_x1 = max(0, x1 - FACE_PADDING)
            pad_y1 = max(0, y1 - FACE_PADDING)
            pad_x2 = min(frame_w, x2 + FACE_PADDING)
            pad_y2 = min(frame_h, y2 + FACE_PADDING)
            face_img = frame[pad_y1:pad_y2, pad_x1:pad_x2]

            if face_img.size == 0:
                print(f"[{camera_id}] Warning: Cropped face image is empty. Skipping.")
                continue

            current_embedding = None
            try:
                embedding_objs = DeepFace.represent(
                    img_path=face_img, model_name=DEEPFACE_MODEL, enforce_detection=False,
                    detector_backend='opencv', align=True
                )
                if embedding_objs:
                    current_embedding = embedding_objs[0]['embedding']
            except Exception as e:
                print(f"[{camera_id}] Error generating embedding: {e}")
                cv2.putText(display_frame, "Emb Err", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            if current_embedding is not None:
                online_match = find_online_match(current_embedding)

                if online_match:
                    display_text = f"{online_match['name']} ({online_match['status']})"
                    text_color = (0, 255, 0) if online_match['status'].lower() == 'allowed' else (0, 165, 255)
                    cv2.putText(display_frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                    person_mongo_id = online_match.get('_id')
                    if person_mongo_id:
                        try:
                            _, buffer = cv2.imencode('.jpg', face_img)
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            captured_image_data = f"data:image/jpeg;base64,{jpg_as_text}"
                            log_payload = {
                                "personId": person_mongo_id,
                                "location": camera_id,
                                "capturedImage": captured_image_data
                            }
                            response = requests.post(DETECTIONS_API_ENDPOINT, json=log_payload, timeout=10)
                            response.raise_for_status()
                            print(f"[{camera_id}] Successfully sent detection log for {online_match['name']} to backend.")
                        except requests.exceptions.RequestException as req_err:
                            print(f"[{camera_id}] Error sending detection log for {online_match['name']}: {req_err}")
                        except Exception as e:
                            print(f"[{camera_id}] General error during detection log sending: {e}")
                    else:
                        print(f"[{camera_id}] Cannot send log: MongoDB _id missing for matched person {online_match.get('name')}")
                else:
                    cv2.putText(display_frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    is_duplicate = is_local_duplicate(current_embedding)

                    if not is_duplicate:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = os.path.join(CAPTURE_DIR, f"face_{camera_id}_{timestamp}.jpg")
                        try:
                            cv2.imwrite(filename, face_img)
                            camera_state['faces_captured'] += 1
                            print(f"[{camera_id}] Captured NEW Unknown face #{camera_state['faces_captured']} - {os.path.basename(filename)}")

                            new_entry = {'path': filename, 'embedding': current_embedding}
                            face_database.append(new_entry)
                            save_database()
                        except Exception as save_err:
                            print(f"[{camera_id}] Error during face saving/DB update: {save_err}")
                    else:
                        camera_state['duplicates_skipped'] += 1

    cv2.putText(display_frame, f"Cam: {camera_id}", (10, frame.shape[0] - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Unique: {camera_state['faces_captured']}", (10, frame.shape[0] - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Dupes: {camera_state['duplicates_skipped']}", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return display_frame, camera_state

try:
    while True:
        frame_webcam = None
        frame_ipcam = None

        if webcam_available:
            ret_webcam, frame_webcam = cap_webcam.read()
            if not ret_webcam:
                print("Warning: Failed to capture frame from Webcam!")
                frame_webcam = None

        if ipcam_available:
            ret_ipcam, frame_ipcam = cap_ipcam.read()
            if not ret_ipcam:
                print("Warning: Failed to capture frame from IP Camera!")
                frame_ipcam = None

        if frame_webcam is not None:
            processed_frame_webcam, camera_states['webcam'] = process_frame(
                frame_webcam, 'webcam', camera_states['webcam']
            )
        else:
            processed_frame_webcam = np.zeros((480, 640, 3), dtype=np.uint8)

        if frame_ipcam is not None:
            processed_frame_ipcam, camera_states['ipcam'] = process_frame(
                frame_ipcam, 'ipcam', camera_states['ipcam']
            )
        else:
            processed_frame_ipcam = np.zeros((480, 640, 3), dtype=np.uint8)

        h_webcam = processed_frame_webcam.shape[0]
        h_ipcam = processed_frame_ipcam.shape[0]

        if h_webcam != h_ipcam:
            target_height = min(h_webcam, h_ipcam)
            processed_frame_webcam = cv2.resize(processed_frame_webcam, (int(processed_frame_webcam.shape[1] * target_height / h_webcam), target_height))
            processed_frame_ipcam = cv2.resize(processed_frame_ipcam, (int(processed_frame_ipcam.shape[1] * target_height / h_ipcam), target_height))

        combined_frame = cv2.hconcat([processed_frame_webcam, processed_frame_ipcam])

        cv2.imshow('Face Detection', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user")
            break

finally:
    print("Cleaning up...")
    if webcam_available and cap_webcam.isOpened():
        cap_webcam.release()
        print("Webcam released.")
    if ipcam_available and cap_ipcam.isOpened():
        cap_ipcam.release()
        print("IP Camera released.")
    cv2.destroyAllWindows()
    print(f"Session summary:")
    print(f"- Webcam Unique: {camera_states['webcam']['faces_captured']}, Duplicates: {camera_states['webcam']['duplicates_skipped']}")
    print(f"- IP Cam Unique: {camera_states['ipcam']['faces_captured']}, Duplicates: {camera_states['ipcam']['duplicates_skipped']}")
    print(f"Images saved to: {os.path.abspath(CAPTURE_DIR)}")
