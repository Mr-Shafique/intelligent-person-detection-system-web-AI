import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
import threading
from threading import Lock
import requests
import traceback
from deepface import DeepFace
import base64
from scipy.spatial.distance import cosine, euclidean

# --- Configuration ---
DEEPFACE_MODEL = "VGG-Face"
DEEPFACE_METRIC = "cosine"
MATCH_THRESHOLD = 0.40
BACKEND_URL = "http://localhost:5000" # Your Node.js backend URL

RECOGNITION_DATA_ENDPOINT = f"{BACKEND_URL}/api/persons"

# --- Global Variables ---
known_faces_db = [] # Stores {'name': str, 'cmsId': str, 'embedding': list}
recognition_results = {} # Stores latest result per camera: {'cam_name': {'label': '...', 'box': (x1,y1,x2,y2), 'timestamp': ...}}
recognition_lock = Lock() # Lock for accessing recognition_results
last_recognition_start_time = {} # Track last start time per camera
RECOGNITION_INTERVAL = 1.0 # Minimum seconds between recognition attempts per camera

# --- Function to Load Known Faces from Backend ---
def load_known_faces():
    global known_faces_db
    print(f"[DEBUG] Entering load_known_faces function.") # Added Debug
    print(f"Attempting to load known faces from backend API: {RECOGNITION_DATA_ENDPOINT}")
    try:
        response = requests.get(RECOGNITION_DATA_ENDPOINT, timeout=15)
        print(f"[DEBUG] API Response Status Code: {response.status_code}") # Added Debug
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        persons_data = response.json()
        print(f"[DEBUG] Raw data received from API: {persons_data}") # Added Debug
        print(f"Received data for {len(persons_data)} persons from backend.")

        if not persons_data: # Added check for empty list
            print("[DEBUG] API returned an empty list of persons.")

        loaded_db = []
        for i, person in enumerate(persons_data):
            print(f"[DEBUG] Processing person index {i}: {person}") # Added Debug
            name = person.get('name')
            cmsId = person.get('cmsId')
            # --- Use 'image' field from backend response ---            
            base64_image_data = person.get('image') 
            # ---------------------------------------------

            if not name or not cmsId or not base64_image_data:
                print(f"[DEBUG] Skipping person index {i} due to missing data.") # Added Debug
                print(f"Skipping person due to missing data: {person}")
                continue

            print(f"Processing {name} (CMS: {cmsId}) - Received Base64 data.") # Existing print

            try:
                if ',' in base64_image_data:
                    base64_image_data = base64_image_data.split(',', 1)[1]
                image_bytes = base64.b64decode(base64_image_data)

                image_data = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"[DEBUG] Failed to decode image for person index {i}.") # Added Debug
                    print(f"Warning: Could not decode Base64 image data for {name}")
                    continue

                embedding_objs = DeepFace.represent(
                    img_path=img,
                    model_name=DEEPFACE_MODEL,
                    enforce_detection=True,
                    detector_backend='opencv'
                )

                if embedding_objs:
                    embedding = embedding_objs[0]['embedding']
                    loaded_db.append({
                        'name': name,
                        'cmsId': cmsId,
                        'embedding': embedding
                    })
                    print(f"[DEBUG] Successfully processed person index {i}.") # Added Debug
                    print(f"--> Successfully generated embedding for {name}")
                else:
                    print(f"[DEBUG] No face detected for person index {i}.") # Added Debug
                    print(f"Warning: No face detected in Base64 image for {name}")

            except Exception as e:
                print(f"[DEBUG] Exception during processing person index {i}: {e}") # Added Debug
                print(f"Error processing Base64 image for {name}: {e}")

        known_faces_db = loaded_db
        print(f"Finished loading. Successfully processed {len(known_faces_db)} known faces.")

    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] RequestException occurred.") # Added Debug
        print(f"Error fetching data from backend API ({RECOGNITION_DATA_ENDPOINT}): {e}")
    except Exception as e:
        print(f"[DEBUG] Generic Exception occurred in load_known_faces.") # Added Debug
        print(f"An unexpected error occurred during loading known faces: {e}")
        traceback.print_exc()
    print(f"[DEBUG] Exiting load_known_faces function.") # Added Debug

# --- Recognition Function (to be run in a thread) ---
def recognize_face_thread(face_crop, cam_name, box_coords):
    global recognition_results, recognition_lock, known_faces_db
    try:
        detected_embedding_objs = DeepFace.represent(
            img_path=face_crop,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            detector_backend='skip'
        )

        if not detected_embedding_objs:
            return

        detected_embedding = detected_embedding_objs[0]['embedding']
        match_found = False
        matched_name = "Unknown"
        matched_cmsId = "N/A"
        min_distance = float('inf')

        for known_face in known_faces_db:
            known_embedding = known_face['embedding']
            distance = float('inf')
            if DEEPFACE_METRIC == 'cosine':
                distance = cosine(detected_embedding, known_embedding)
            elif DEEPFACE_METRIC == 'euclidean':
                distance = euclidean(detected_embedding, known_embedding)

            if distance < min_distance and distance < MATCH_THRESHOLD:
                min_distance = distance
                matched_name = known_face['name']
                matched_cmsId = known_face['cmsId']
                match_found = True

        label = f"{matched_name} ({matched_cmsId})" if match_found else "Unknown"
        result_info = {'label': label, 'box': box_coords, 'timestamp': time.time()}

        with recognition_lock:
            recognition_results[cam_name] = result_info

        if match_found:
            print(f"Recognition Thread ({cam_name}): Match found - {label} (Distance: {min_distance:.4f})")

    except Exception as e:
        pass

# Thread class for reading video frames
class VideoCaptureThread:
    def __init__(self, src=0, name="CameraThread", backend=None):
        self.src = src
        self.name = name
        if backend:
            self.cap = cv2.VideoCapture(self.src, backend)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.src} for {self.name}")
            self.running = False
            self.latest_frame = None
        else:
            print(f"Successfully opened video source {self.src} for {self.name}")
            self.running = True
            self.latest_frame = None
            self.thread = threading.Thread(target=self._reader, name=self.name, daemon=True)
            self.thread.start()
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue
            self.latest_frame = frame
            time.sleep(0.01)
    def read(self):
        return self.latest_frame
    def is_running(self):
        return self.running and self.cap.isOpened()
    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        print(f"Released video source {self.name}")

# Camera sources
laptop_cam_src = 0
ip_cam_src = "http://10.102.138.78:8080/video"

model = YOLO('yolov8n-face.pt')

# --- Load known faces BEFORE starting cameras ---
load_known_faces()
# ----------------------------------------------

# --- Start camera threads AFTER loading known faces ---
laptop_thread = VideoCaptureThread(src=laptop_cam_src, name="LaptopCam")
ip_cam_thread = VideoCaptureThread(src=ip_cam_src, name="IPCam", backend=cv2.CAP_FFMPEG)
# ---------------------------------------------------

print("Starting main loop. Press 'q' to quit.")

try:
    while True:
        frames_to_process = {}
        if laptop_thread.is_running():
            frames_to_process["LaptopCam"] = laptop_thread.read()
        if ip_cam_thread.is_running():
            frames_to_process["IPCam"] = ip_cam_thread.read()

        processed_display_frames = {}
        current_time = time.time() # Get current time once per loop iteration

        for cam_name, frame in frames_to_process.items():
            if frame is None:
                continue

            processed_frame = frame.copy()

            # --- Face Detection ---
            results = model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

            # --- Get latest recognition result for display ---
            latest_label = "Unknown"
            latest_box = None
            with recognition_lock:
                if cam_name in recognition_results:
                    latest_label = recognition_results[cam_name]['label']
                    latest_box = recognition_results[cam_name]['box']

            # --- Process Detected Boxes ---
            face_processed_this_frame = False # Flag to process only one face per frame if desired
            for box in boxes:
                # --- Check Time Interval ---
                last_start = last_recognition_start_time.get(cam_name, 0)
                if current_time - last_start < RECOGNITION_INTERVAL:
                    # If interval not passed, still draw box but skip starting new recognition
                    color = (0, 255, 0) if latest_label != "Unknown" else (0, 0, 255)
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, latest_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    continue # Skip to next box if time interval not met

                # --- Process Face (if interval passed) ---
                x1, y1, x2, y2 = map(int, box[:4])
                if x2 <= x1 or y2 <= y1: continue

                current_face_img = frame[y1:y2, x1:x2]
                if current_face_img.size == 0: continue

                # --- Start Recognition in Background Thread ---
                print(f"Starting recognition thread for {cam_name} at {current_time:.2f}") # Debug print
                last_recognition_start_time[cam_name] = current_time # Update last start time
                recognition_thread = threading.Thread(
                    target=recognize_face_thread,
                    args=(current_face_img.copy(), cam_name, (x1, y1, x2, y2)),
                    daemon=True
                )
                recognition_thread.start()

                # --- Draw Box and Display LAST KNOWN Label ---
                color = (0, 255, 0) if latest_label != "Unknown" else (0, 0, 255)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, latest_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Optional: Break after processing the first face within the interval
                break # Process only the first detected face that meets the time criteria


            processed_display_frames[cam_name] = processed_frame

        # --- Display Frames ---
        if "LaptopCam" in processed_display_frames and "IPCam" in processed_display_frames:
            h1, w1 = processed_display_frames["LaptopCam"].shape[:2]
            h2, w2 = processed_display_frames["IPCam"].shape[:2]
            display_height = 360
            try:
                laptop_resized = cv2.resize(processed_display_frames["LaptopCam"], (int(w1 * display_height / h1), display_height)) if h1 > 0 else np.zeros((display_height, 300, 3), dtype=np.uint8)
                ipcam_resized = cv2.resize(processed_display_frames["IPCam"], (int(w2 * display_height / h2), display_height)) if h2 > 0 else np.zeros((display_height, 300, 3), dtype=np.uint8)
                combined = cv2.hconcat([laptop_resized, ipcam_resized])
                cv2.namedWindow('Laptop (Left) & IP Camera (Right)', cv2.WINDOW_NORMAL)
                cv2.imshow('Laptop (Left) & IP Camera (Right)', combined)
            except Exception as e:
                print(f"Error resizing/displaying combined frames: {e}")
        elif "LaptopCam" in processed_display_frames:
            try:
                cv2.namedWindow('Laptop Camera', cv2.WINDOW_NORMAL)
                cv2.imshow('Laptop Camera', processed_display_frames["LaptopCam"])
            except Exception as e:
                print(f"Error displaying LaptopCam frame: {e}")
        elif "IPCam" in processed_display_frames:
            try:
                cv2.namedWindow('IP Camera', cv2.WINDOW_NORMAL)
                cv2.imshow('IP Camera', processed_display_frames["IPCam"])
            except Exception as e:
                print(f"Error displaying IPCam frame: {e}")

        # --- Quit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received.")
            break

        # --- Check Camera Threads ---
        if not laptop_thread.is_running() and not ip_cam_thread.is_running():
            print("Both camera threads seem to have stopped.")
            time.sleep(1)
            if not laptop_thread.is_running() and not ip_cam_thread.is_running():
                break

finally:
    print("Releasing resources...")
    laptop_thread.release()
    ip_cam_thread.release()
    cv2.destroyAllWindows()
    print("Cleanup finished.")
