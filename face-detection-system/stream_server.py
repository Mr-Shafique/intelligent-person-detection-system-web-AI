import cv2

def count_available_cameras():
    index = 0
    arr = []
    # Try to open cameras and add their indices to the list
    # This loop might be slow if many non-existent indices are checked.
    # A common approach is to check a reasonable number, e.g., up to 10.
    for i in range(10): # Check up to 10 potential camera indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            arr.append(i)
            cap.release()
    return len(arr), arr

num_cameras, camera_indices = count_available_cameras()
print(f"Number of available cameras: {num_cameras}")

if num_cameras > 0:
    print(f"Available camera indices: {camera_indices}")
    
    caps = []
    for index in camera_indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            caps.append(cap)
        else:
            print(f"Warning: Could not open camera at index {index} even after detection.")

    if not caps:
        print("No cameras could be opened for display.")
    else:
        while True:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Camera Index {camera_indices[i]}", frame)
                else:
                    # Optionally, display a black frame or message if a camera fails mid-stream
                    print(f"Warning: Could not read frame from camera index {camera_indices[i]}")
                    # Create a black frame as a placeholder
                    black_frame = cv2.UMat(480, 640, cv2.CV_8UC3) # Or use np.zeros
                    black_frame[:] = 0
                    cv2.imshow(f"Camera Index {camera_indices[i]}", black_frame)


            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release all captures and destroy all windows
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
else:
    print("No cameras detected to display.")