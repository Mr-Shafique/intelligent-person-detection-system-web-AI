import cv2
import os
from ultralytics import YOLO
import time
import numpy as np

# Load YOLOv8 model
# Ensure 'yolov8n-face.pt' is in the same directory as camera.py or provide the full path
try:
    model = YOLO('yolov8n-face.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolov8n-face.pt' is in the correct location.")
    exit()

# Create directory to save captured frames with faces
output_dir = 'capturedfaces'
os.makedirs(output_dir, exist_ok=True)

# Open the webcam
webcam = cv2.VideoCapture(0) # Or try other indices like 1, 2 if 0 doesn't work

# Open the IP camera
ip_camera_url = "http://10.102.138.81:8080/video"  # Replace with your IP camera URL
ip_camera = cv2.VideoCapture(ip_camera_url)

# Check initial availability
webcam_available = webcam.isOpened()
if not webcam_available:
    print("Warning: Could not open the webcam.")

ip_camera_available = ip_camera.isOpened()
if not ip_camera_available:
    print("Warning: Could not connect to the IP camera.")

if not webcam_available and not ip_camera_available:
    print("Error: Neither webcam nor IP camera is available. Exiting.")
    if webcam.isOpened(): webcam.release()
    if ip_camera.isOpened(): ip_camera.release()
    cv2.destroyAllWindows()
    exit()

print("Press 'q' to quit.")

webcam_frame_count = 0
ipcam_frame_count = 0
PROCESS_EVERY_N_FRAMES = 1 # Process every Nth frame for detection

# Store initial intended state for display layout
initial_webcam_intended = webcam_available
initial_ip_camera_intended = ip_camera_available

FACE_CLASS_ID = 0 # Assuming class 0 is 'face' for yolov8n-face.pt

while True:
    webcam_display_frame = None
    ipcam_display_frame = None

    # Process webcam if it's currently marked available
    if webcam_available:
        ret_webcam, frame_webcam = webcam.read()
        if ret_webcam:
            webcam_frame_count += 1
            # Resize for detection and consistent display size
            resized_webcam_for_detection = cv2.resize(frame_webcam, (640, 480))
            webcam_display_frame = resized_webcam_for_detection.copy() # Prepare frame for display & drawing

            if webcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_webcam = model(resized_webcam_for_detection, conf=0.5, verbose=False)
                
                found_face_in_webcam_frame = False
                if results_webcam and results_webcam[0].boxes.data.numel() > 0:
                    for box in results_webcam[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                        conf = box.conf[0].item()             # Confidence score
                        cls = int(box.cls[0].item())          # Class ID

                        if cls == FACE_CLASS_ID:
                            found_face_in_webcam_frame = True
                            # Draw bounding box on the display frame
                            cv2.rectangle(webcam_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Face: {conf:.2f}"
                            cv2.putText(webcam_display_frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # No need to break if you want to draw all detected faces
                
                if found_face_in_webcam_frame:
                    timestamp = int(time.time() * 1000)
                    frame_filename = os.path.join(output_dir, f"webcam_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, frame_webcam) # Save original high-res frame
        else:
            print("Warning: Webcam frame read failed. Marking as unavailable.")
            webcam.release() # Release it properly
            webcam_available = False
            webcam_display_frame = None # Ensure it's None if read failed

    # Process IP camera if it's currently marked available
    if ip_camera_available:
        ret_ip, frame_ip = ip_camera.read()
        if ret_ip:
            ipcam_frame_count += 1
            resized_ip_for_detection = cv2.resize(frame_ip, (640, 480))
            ipcam_display_frame = resized_ip_for_detection.copy()

            if ipcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_ip = model(resized_ip_for_detection, conf=0.5, verbose=False)

                found_face_in_ipcam_frame = False
                if results_ip and results_ip[0].boxes.data.numel() > 0:
                    for box in results_ip[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        
                        if cls == FACE_CLASS_ID:
                            found_face_in_ipcam_frame = True
                            cv2.rectangle(ipcam_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Face: {conf:.2f}"
                            cv2.putText(ipcam_display_frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if found_face_in_ipcam_frame:
                    timestamp = int(time.time() * 1000)
                    frame_filename = os.path.join(output_dir, f"ip_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_filename, frame_ip) # Save original high-res frame
        else:
            print("Warning: IP camera frame read failed. Marking as unavailable.")
            ip_camera.release() # Release it properly
            ip_camera_available = False
            ipcam_display_frame = None # Ensure it's None if read failed

    # If both cameras have become unavailable, exit the loop
    if not webcam_available and not ip_camera_available:
        print("Error: Both cameras are now unavailable. Exiting.")
        break

    # Construct the frame for display
    display_frame_final = None
    window_title = "Camera Feed"
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Standard black frame

    # Determine current frames for display, using black_frame if a camera is down or frame is None
    current_webcam_display = webcam_display_frame if webcam_available and webcam_display_frame is not None else black_frame
    current_ipcam_display = ipcam_display_frame if ip_camera_available and ipcam_display_frame is not None else black_frame

    if initial_webcam_intended and initial_ip_camera_intended:
        # Ensure both frames are valid for hstack, even if one is black
        display_frame_final = np.hstack((current_webcam_display, current_ipcam_display))
        window_title = 'Camera Feeds (Webcam | IP Cam)'
    elif initial_webcam_intended:
        display_frame_final = current_webcam_display
        window_title = 'Webcam Feed'
    elif initial_ip_camera_intended:
        display_frame_final = current_ipcam_display
        window_title = 'IP Camera Feed'
    
    if display_frame_final is not None:
        cv2.imshow(window_title, display_frame_final)
    else:
        # This case might be hit if no cameras were ever intended, or some logic error
        # Create a small black frame to keep imshow happy if it's called with None
        if not (initial_webcam_intended or initial_ip_camera_intended):
             cv2.imshow("No Camera", np.zeros((100,100,3), dtype=np.uint8))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
if webcam.isOpened():
    webcam.release()
if ip_camera.isOpened():
    ip_camera.release()
cv2.destroyAllWindows()