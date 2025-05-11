import cv2
import os
from ultralytics import YOLO
import time
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n-face.pt')

# Create directory to save captured frames with faces
output_dir = 'capturedfaces'
os.makedirs(output_dir, exist_ok=True)

# Open the webcam
webcam = cv2.VideoCapture(0)

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
    exit()

print("Press 'q' to quit.")

webcam_frame_count = 0
ipcam_frame_count = 0
PROCESS_EVERY_N_FRAMES = 1 # Process every Nth frame for detection (e.g., 1 means every frame)

# Store initial intended state for display layout
initial_webcam_intended = webcam_available
initial_ip_camera_intended = ip_camera_available

while True:
    webcam_display_frame = None
    ipcam_display_frame = None

    # Process webcam if it's currently marked available
    if webcam_available:
        ret_webcam, frame_webcam = webcam.read()
        if ret_webcam:
            webcam_frame_count += 1
            # Resize for detection and consistent display size
            resized_webcam = cv2.resize(frame_webcam, (640, 480))
            webcam_display_frame = resized_webcam  # Always prepare for display

            if webcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_webcam = model(resized_webcam, conf=0.75) # Detect on resized frame
                for result in results_webcam[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = result
                    if int(cls) == 0:  # Class 0 is 'face'
                        timestamp = int(time.time() * 1000)
                        frame_filename = os.path.join(output_dir, f"webcam_frame_{timestamp}.jpg")
                        cv2.imwrite(frame_filename, frame_webcam) # Save original frame
                        break  # Process only the first detected face for saving
        else:
            print("Warning: Webcam frame read failed. Marking as unavailable.")
            webcam_available = False

    # Process IP camera if it's currently marked available
    if ip_camera_available:
        ret_ip, frame_ip = ip_camera.read()
        if ret_ip:
            ipcam_frame_count += 1
            # Resize for detection and consistent display size
            resized_ip = cv2.resize(frame_ip, (640, 480))
            ipcam_display_frame = resized_ip  # Always prepare for display

            if ipcam_frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results_ip = model(resized_ip, conf=0.75) # Detect on resized frame
                for result in results_ip[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = result
                    if int(cls) == 0:  # Class 0 is 'face'
                        timestamp = int(time.time() * 1000)
                        frame_filename = os.path.join(output_dir, f"ip_frame_{timestamp}.jpg")
                        cv2.imwrite(frame_filename, frame_ip) # Save original frame
                        break  # Process only the first detected face for saving
        else:
            print("Warning: IP camera frame read failed. Marking as unavailable.")
            ip_camera_available = False

    # If both cameras have become unavailable, exit the loop
    if not webcam_available and not ip_camera_available:
        print("Error: Both cameras are now unavailable. Exiting.")
        break

    # Construct the frame for display
    display_frame_final = None
    window_title = "Camera Feed"
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Determine current frames for display, using black_frame if a camera is down
    current_webcam_display = webcam_display_frame if webcam_available and webcam_display_frame is not None else black_frame
    current_ipcam_display = ipcam_display_frame if ip_camera_available and ipcam_display_frame is not None else black_frame

    if initial_webcam_intended and initial_ip_camera_intended:
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
        # This case should not be reached if at least one camera was initially intended
        # and the loop hasn't broken due to both becoming unavailable.
        # If it is, it implies an issue with initial state or loop break logic.
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
if webcam.isOpened(): # Check again before release, in case it was never opened or already released
    webcam.release()
if ip_camera.isOpened():
    ip_camera.release()
cv2.destroyAllWindows()