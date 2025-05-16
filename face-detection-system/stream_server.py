from flask import Flask, Response
import cv2
import time
import threading
import numpy as np
import os
import pickle
import datetime
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine

YOLO_MODEL_PATH = 'yolov8n-face.pt'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
RECOGNITION_THRESHOLD = 0.6
PROCESS_EVERY_N_FRAMES = 2
FACE_CLASS_ID = 0

LINE_A_Y = 200
LINE_B_Y = 300

app = Flask(__name__)
latest_frame = [None]  # This will be updated by the video processing thread

# Load YOLO model
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load known embeddings
known_face_embeddings = []
def load_known_embeddings(file_path=EMBEDDINGS_FILE):
    global known_face_embeddings
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                known_face_embeddings = pickle.load(f)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            known_face_embeddings = []
    else:
        known_face_embeddings = []
load_known_embeddings()

def create_face_embedding_live(face_roi):
    try:
        embedding_obj = DeepFace.represent(face_roi, model_name="Facenet", enforce_detection=False, detector_backend='skip', align=True)
        if embedding_obj and len(embedding_obj) > 0 and "embedding" in embedding_obj[0]:
            return embedding_obj[0]["embedding"]
    except Exception:
        pass
    return None

def find_match(live_face_roi, known_embeddings_data, threshold):
    live_embedding = create_face_embedding_live(live_face_roi)
    if not live_embedding:
        return "Unknown", None, None, float('inf')
    best_match_name = "Unknown"
    best_match_cmsId = None
    best_match_status = None
    min_distance = float('inf')
    if not known_embeddings_data:
        return best_match_name, best_match_cmsId, best_match_status, min_distance
    v1 = np.array(live_embedding)
    if v1.ndim > 1: v1 = v1.flatten()
    for known_entry in known_embeddings_data:
        known_embedding_vector = known_entry.get("imageEmbedding")
        if not known_embedding_vector:
            continue
        v2 = np.array(known_embedding_vector)
        if v2.ndim > 1: v2 = v2.flatten()
        if v1.shape != v2.shape:
            continue
        try:
            distance = cosine(v1, v2)
        except Exception:
            continue
        if distance < min_distance:
            min_distance = distance
            if distance < threshold:
                best_match_name = known_entry.get("name", "ErrorName")
                best_match_cmsId = known_entry.get("cmsId", "ErrorID")
                best_match_status = known_entry.get("status", "ErrorStatus")
            else:
                best_match_name = "Unknown"
                best_match_cmsId = None
                best_match_status = None
    return best_match_name, best_match_cmsId, best_match_status, min_distance

def video_processing_loop():
    global latest_frame
    ip_camera_url = "http://10.102.139"  # Replace with your actual IP camera stream URL
    print(f"[INFO] Opening IP camera at {ip_camera_url} ...")
    ip_camera = cv2.VideoCapture(ip_camera_url)
    if not ip_camera.isOpened():
        print(f"[ERROR] Could not open IP camera at {ip_camera_url}. Check the URL and camera status.")
        return
    frame_count = 0
    while True:
        ret, frame_ip_original = ip_camera.read()
        if not ret:
            print("[WARN] Failed to read frame from IP camera. Retrying in 0.1s...")
            time.sleep(0.1)
            continue
        frame_count += 1
        resized_ip_for_detection = cv2.resize(frame_ip_original, (640, 480))
        ipcam_display_frame = resized_ip_for_detection.copy()
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results_ip = model(resized_ip_for_detection, conf=0.5, verbose=False)
            if results_ip and results_ip[0].boxes.data.numel() > 0:
                for box in results_ip[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    face_roi = frame_ip_original[y1:y2, x1:x2]
                    name, cmsId, status, dist = find_match(face_roi, known_face_embeddings, RECOGNITION_THRESHOLD)
                    color = (255, 0, 0) if cmsId else (0, 0, 255)
                    cv2.rectangle(ipcam_display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name}" if name != "Unknown" else "Unknown"
                    cv2.putText(ipcam_display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # Draw lines for IN/OUT reference (optional, for visual consistency)
        cv2.line(ipcam_display_frame, (0, LINE_A_Y), (640, LINE_A_Y), (0, 255, 0), 2)
        cv2.putText(ipcam_display_frame, "A", (5, LINE_A_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.line(ipcam_display_frame, (0, LINE_B_Y), (640, LINE_B_Y), (0, 0, 255), 2)
        cv2.putText(ipcam_display_frame, "B", (5, LINE_B_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        latest_frame[0] = ipcam_display_frame
        time.sleep(0.03)  # ~30 FPS

threading.Thread(target=video_processing_loop, daemon=True).start()

def gen_frames():
    while True:
        if latest_frame[0] is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame[0])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)