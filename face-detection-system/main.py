import cv2
import os
import time
import numpy as np
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import pickle
import requests # Added requests import
import traceback # Added traceback import

# --- Configuration ---
DEEPFACE_MODEL = "VGG-Face"
DEEPFACE_METRIC = "cosine"
MATCH_THRESHOLD = 0.65  # Threshold for local duplicate check

# Local Capture Config
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_faces")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_embeddings.pkl") # Local captures
CAPTURE_INTERVAL = 0.5
FACE_PADDING = 30

# Online API Config
ONLINE_EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_face_embeddings.pkl") # Online data
API_ENDPOINT = "http://localhost:5000/api/persons"
BACKEND_URL = "http://localhost:5000" # Base URL for images

# Storage for face embeddings
face_database = []  # Local captures: {path: str, embedding: np.array}
online_face_database = [] # Online data: {name: str, cmsId: str, status: str, embedding: np.array}

# Create output directory
os.makedirs(CAPTURE_DIR, exist_ok=True)
print(f"Images will be saved to: {os.path.abspath(CAPTURE_DIR)}")
print(f"Embeddings database file: {os.path.abspath(EMBEDDINGS_FILE)}") # Added path confirmation
print(f"Online embeddings database file: {os.path.abspath(ONLINE_EMBEDDINGS_FILE)}") # Added path confirmation

# Function to save the LOCAL database to a file
def save_database():
    try:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(face_database, f)
        # print(f"[Debug] Database saved with {len(face_database)} entries.") # Optional debug print
    except Exception as e:
        print(f"Error saving database to {EMBEDDINGS_FILE}: {e}")

# Function to save the ONLINE database to a file
def save_online_database():
    # Print the state of the list JUST BEFORE saving - Ensure this is commented out
    # print(f"[SAVE_ONLINE] Data to be saved (Size: {len(online_face_database)}): {online_face_database}")
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
    file_existed_initially = os.path.exists(EMBEDDINGS_FILE) # Check if file exists before loading

    # 1. Try loading from pickle file
    if file_existed_initially:
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                face_database = pickle.load(f)
            print(f"Loaded {len(face_database)} embeddings from {EMBEDDINGS_FILE}")
            # Populate known_paths from the loaded database
            for entry in face_database:
                known_paths.add(entry.get('path'))
        except Exception as e:
            print(f"Error loading database from {EMBEDDINGS_FILE}: {e}. Initializing empty database.")
            face_database = []
            # Consider the file corrupt/unusable, treat as if it didn't exist for saving purposes
            file_existed_initially = False
    else:
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Initializing empty database.")
        face_database = []

    # 2. Scan CAPTURE_DIR for images not in the loaded database
    print(f"Scanning {CAPTURE_DIR} for new faces...")
    new_faces_found = 0
    for filename in os.listdir(CAPTURE_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        filepath = os.path.join(CAPTURE_DIR, filename)

        # Check if this image path is already in our database
        if filepath in known_paths:
            continue # Skip if already loaded/processed

        # If not known, process it
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
                known_paths.add(filepath) # Add to known paths immediately
                database_updated = True # Mark database as updated
                if new_faces_found % 10 == 0:
                     print(f"Added {new_faces_found} new faces to database...")
            else:
                 print(f"Warning: No embedding generated for new file {filename}")

        except Exception as e:
            print(f"Error processing new image {filepath}: {e}")

    print(f"Found and processed {new_faces_found} new faces from directory scan.")

    # 3. Save the database if it was updated OR if the file didn't exist initially
    if database_updated:
        print("Database was updated by scan, saving changes...")
        save_database()
    elif not file_existed_initially:
        print(f"Embeddings file did not exist, creating it now (even if empty)...")
        save_database() # Save even if empty to create the file

    print(f"Database ready with {len(face_database)} total entries.")


# Function to load ONLINE face embeddings from API and file (Modified to always refresh from API)
def load_online_faces():
    global online_face_database
    # Always start with an empty database for online faces to refresh completely
    online_face_database = []
    # Confirm the list is empty right after initialization
    print(f"[LOAD_ONLINE] Initialized empty online face database. Current size: {len(online_face_database)}")

    # 1. Fetch data from API
    print(f"[LOAD_ONLINE] Fetching person data from API: {API_ENDPOINT}")
    persons_data = [] # Default to empty list
    try:
        response = requests.get(API_ENDPOINT, timeout=20)
        print(f"[LOAD_ONLINE] API response status code: {response.status_code}") # Debug status code
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        persons_data = response.json()
        print(f"[LOAD_ONLINE] Received data for {len(persons_data)} persons from API.")

        if not isinstance(persons_data, list):
             print(f"[LOAD_ONLINE] Error: API response is not a list. Received type: {type(persons_data)}")
             persons_data = [] # Treat as empty if format is wrong

    except requests.exceptions.RequestException as e:
        print(f"[LOAD_ONLINE] FATAL: Error fetching data from API ({API_ENDPOINT}): {e}")
    except Exception as e:
        print(f"[LOAD_ONLINE] FATAL: Error processing API response: {e}")
        traceback.print_exc()


    # 2. Process ALL persons from API response
    api_faces_processed_count = 0
    image_fields = ['frontImage', 'leftImage', 'rightImage']

    print(f"[LOAD_ONLINE] Starting processing for {len(persons_data)} persons.")
    for i, person in enumerate(persons_data):
        cmsId = person.get('cmsId')
        name = person.get('name')
        status = person.get('status', 'unknown') # Default status if missing
        print(f"[LOAD_ONLINE] Person {i+1}/{len(persons_data)}: ID={cmsId}, Name={name}") # Debug each person

        if not cmsId or not name:
            print(f"  [LOAD_ONLINE] Warning: Skipping person due to missing cmsId or name: {person}")
            continue

        person_embeddings_added = 0
        valid_image_found = False # Track if any valid image URL exists for this person
        for field in image_fields:
            relative_image_path = person.get(field)
            if not relative_image_path:
                continue

            valid_image_found = True # Mark that at least one image URL exists
            image_url = f"{BACKEND_URL}{relative_image_path}"

            try:
                img_response = requests.get(image_url, timeout=10)
                img_response.raise_for_status()
                image_data = np.frombuffer(img_response.content, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"    [LOAD_ONLINE] Warning: Could not decode image from {image_url}")
                    continue

                # Generate embedding (enforce detection for known API faces)
                embedding_objs = DeepFace.represent(
                    img_path=img,
                    model_name=DEEPFACE_MODEL,
                    enforce_detection=True,
                    detector_backend='opencv',
                    align=True
                )

                if embedding_objs:
                    embedding = embedding_objs[0]['embedding']
                    # Append directly to the fresh online_face_database
                    online_face_database.append({
                        'name': name, 'cmsId': cmsId, 'status': status,
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

        # Add more detail if no embeddings were added for this person
        if person_embeddings_added == 0:
             if not valid_image_found:
                 print(f"  [LOAD_ONLINE] -> No embeddings generated for {name} ({cmsId}) because no valid image URLs were found in API data.")
             else:
                 print(f"  [LOAD_ONLINE] -> No embeddings generated for {name} ({cmsId}) despite trying images. Check warnings above (decode/detection failures).")


    print(f"[LOAD_ONLINE] Finished processing API data. Generated {api_faces_processed_count} embeddings this run.")
    # Print the size AGAIN just before calling save
    print(f"[LOAD_ONLINE] Final in-memory online_face_database size before save: {len(online_face_database)}")

    # 3. Always save the newly generated online database, overwriting the old file
    save_online_database() # Call the function with added debug prints

    # Add a final confirmation print AFTER the save call returns
    print(f"[LOAD_ONLINE] Online database refresh process complete. Final in-memory list size: {len(online_face_database)}")


# Function to check if a face already exists in our LOCAL database
def is_duplicate_face(face_img):
    min_distance_found = float('inf') # Track minimum distance for debugging
    matched_file = None # Track which file caused the minimum distance

    try:
        # Generate embedding for detected face from live video
        embedding_objs = DeepFace.represent(
            img_path=face_img,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,  # More permissive for webcam images
            detector_backend='opencv',
            align=True # Added alignment which might improve consistency
        )

        if not embedding_objs:
            # print("[Debug] No embedding generated for current face.")
            return False  # No face found, so can't be a duplicate

        current_embedding = embedding_objs[0]['embedding']

        # Compare with database (loaded from face_embeddings.pkl via load_existing_faces)
        if not face_database: # Check if database is empty
             # print("[Debug] Face database is empty.")
             return False

        # <<< COMPARISON HAPPENS HERE >>>
        for stored_face in face_database: # Iterate through embeddings loaded from .pkl file
            stored_embedding = stored_face['embedding']

            # Calculate similarity using the configured metric
            if DEEPFACE_METRIC == 'cosine':
                # Compare current embedding vs stored embedding
                distance = cosine(current_embedding, stored_embedding)

                # Track the minimum distance found
                if distance < min_distance_found:
                    min_distance_found = distance
                    matched_file = os.path.basename(stored_face['path'])

                # Check if the distance is below the threshold
                if distance < MATCH_THRESHOLD:
                    print(f"[Debug] Duplicate Check: Match found! Current face vs {os.path.basename(stored_face['path'])} -> Distance: {distance:.4f} (Threshold: {MATCH_THRESHOLD})")
                    return True # Found a match from the .pkl file data
        # <<< END OF COMPARISON LOOP >>>

        # If loop completes without returning True, no duplicate was found above threshold
        # Print the minimum distance found for debugging purposes
        if min_distance_found != float('inf'):
             print(f"[Debug] Duplicate Check: No match above threshold. Min distance found: {min_distance_found:.4f} (vs {matched_file})")
        else:
             print("[Debug] Duplicate Check: No similar faces found in DB yet.")

        return False

    except Exception as e:
        print(f"Error checking for duplicate: {e}")
        return False  # If we can't check, assume it's not a duplicate


# Initialize YOLO face detector
print("Loading YOLOv8 face detection model...")
model = YOLO('yolov8n-face.pt')
print("YOLO model loaded.")

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 = default camera (usually webcam)
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit(1)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load existing LOCAL face embeddings
load_existing_faces()

# Load/Refresh ONLINE face embeddings from API/file
load_online_faces() # Call the modified function

# Variables to track last capture time
last_capture_time = 0
faces_captured = 0
duplicates_skipped = 0

print("Starting face detection. Press 'q' to quit.")

try:
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break

        # Make a copy for display
        display_frame = frame.copy()

        # --- Face Detection using YOLO ---
        results = model(frame, verbose=False)
        # Extract boxes in xyxy format
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 and results[0].boxes is not None else []
        # ---------------------------------

        current_time = time.time()

        # Process each detected face
        for box in boxes: # Iterate through YOLO boxes
            x1, y1, x2, y2 = map(int, box[:4]) # Get coordinates
            w = x2 - x1 # Calculate width
            h = y2 - y1 # Calculate height

            # Skip tiny detections (optional)
            if w < 20 or h < 20:
                continue

            # Draw rectangle around face using x1, y1, x2, y2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if enough time has passed since last capture
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                # Add padding around face (but stay within image bounds)
                frame_h, frame_w = frame.shape[:2]
                # Use x1, y1, x2, y2 for padding calculation
                pad_x1 = max(0, x1 - FACE_PADDING)
                pad_y1 = max(0, y1 - FACE_PADDING)
                pad_x2 = min(frame_w, x2 + FACE_PADDING)
                pad_y2 = min(frame_h, y2 + FACE_PADDING)

                # Crop face region with padding
                face_img = frame[pad_y1:pad_y2, pad_x1:pad_x2]

                # Check if this face is already in our database
                is_duplicate = is_duplicate_face(face_img) # Call the updated function

                if not is_duplicate:
                    # Generate unique filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = os.path.join(CAPTURE_DIR, f"face_{timestamp}.jpg")
                    
                    # Save image
                    cv2.imwrite(filename, face_img)
                    faces_captured += 1
                    print(f"Captured NEW face #{faces_captured} - {os.path.basename(filename)}")

                    # Add to our database IN MEMORY
                    new_entry = None
                    try:
                        # Re-generate embedding for the saved image to store
                        embedding_objs_saved = DeepFace.represent(
                            img_path=filename, # Use the saved file path
                            model_name=DEEPFACE_MODEL,
                            enforce_detection=False,
                            detector_backend='opencv',
                            align=True # Added alignment
                        )
                        if embedding_objs_saved:
                            embedding = embedding_objs_saved[0]['embedding']
                            new_entry = {
                                'path': filename,
                                'embedding': embedding
                            }
                            face_database.append(new_entry) # Add to in-memory list
                        else:
                             print(f"Warning: Could not generate embedding for saved file {filename}")
                    except Exception as e:
                        print(f"Error adding face to database: {e}")

                    # Save the updated database to the file IMMEDIATELY
                    if new_entry:
                        save_database()

                    # Feedback on saved image (green text) - Use x1, y1 for position
                    cv2.putText(display_frame, "New Face!", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # It's a duplicate, don't save
                    duplicates_skipped += 1
                    # Use x1, y1 for position
                    cv2.putText(display_frame, "Duplicate", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Update last capture time regardless of duplicate status
                last_capture_time = current_time
            else:
                # Show "wait" message if capturing too frequently (red text) - Use x1, y1 for position
                cv2.putText(display_frame, "Wait...", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display stats on frame
        cv2.putText(display_frame, f"Unique faces: {faces_captured}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Duplicates: {duplicates_skipped}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Face Detection', display_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user")
            break

finally:
    # Release resources
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session summary:")
    print(f"- Unique faces captured: {faces_captured}")
    print(f"- Duplicates skipped: {duplicates_skipped}")
    print(f"Images saved to: {os.path.abspath(CAPTURE_DIR)}")
