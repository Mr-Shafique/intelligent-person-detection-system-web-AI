import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
import time

# Load face embeddings and names
try:
    with open('face_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
        print(type(data))
        print(data)
        # If it's a dict with keys 'embeddings' and 'names'
        known_face_embeddings = data['embeddings']
        known_face_names = data['names']
except FileNotFoundError:
    print("Error: face_embeddings.pkl not found. Please generate it first.")
    exit()

# Initialize face analysis model
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Function to calculate embeddings for a given face
def get_embedding(face_img):
    face = Face(bbox=np.array([0,0,face_img.shape[1],face_img.shape[0]]))
    face.embedding = app.get(face_img)
    return face.embedding

# Function to find the closest match in known embeddings
def recognize_face(face_embedding, threshold=0.5):
    min_distance = 100
    identity = "Unknown"
    for i, known_embedding in enumerate(known_face_embeddings):
        dist = np.sum(np.square(face_embedding - known_embedding))
        if dist < min_distance:
            min_distance = dist
            identity = known_face_names[i]
    
    if min_distance > threshold:
        identity = "Unknown"
    return identity, min_distance

# Open IP camera
ip_camera_url = "http://192.168.220.230:8080/video"  # Replace with your IP camera URL
video_capture = cv2.VideoCapture(ip_camera_url)

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect faces in the frame
    faces = app.get(frame)

    # Loop through each detected face
    if faces:
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Extract the face ROI
            face_roi = frame[y1:y2, x1:x2]

            # Get embedding for the face ROI
            face_embedding = face.embedding

            # Recognize the face
            identity, distance = recognize_face(face_embedding)

            # Draw bounding box and name on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, identity, (x1, y1-10), font,fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()