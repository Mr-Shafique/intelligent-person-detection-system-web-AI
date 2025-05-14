import requests
import json
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from deepface import DeepFace
import os

# Verify if YOLOv8 model file exists
yolo_model_path = "yolov8n-face.pt"
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"YOLOv8 model file not found: {yolo_model_path}")

# Load the YOLOv8 model
model = YOLO(yolo_model_path)

def create_face_embedding(face):
    """
    Create a face embedding using DeepFace.
    """
    try:
        # Generate embedding using DeepFace
        embedding = DeepFace.represent(face, model_name="SFace", enforce_detection=False, detector_backend='skip')
        return embedding[0]["embedding"]  # Return the embedding vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def fetch_and_store_embeddings():
    """
    Fetches person data from an API, processes each image to create embeddings,
    and saves the embeddings to face_embeddings.pkl.
    """
    api_url = "http://localhost:5000/api/persons"
    output_file = "face_embeddings.pkl"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        persons = response.json()

        # Overwrite the face_embeddings.pkl file by clearing old data
        face_embeddings = []

        for person in persons:
            for image_type in ["frontImage", "leftImage", "rightImage"]:
                image_path = person.get(image_type)
                if not image_path:
                    print(f"No {image_type} found for {person['name']}")
                    continue

                image_url = "http://localhost:5000" + image_path  # Construct the full image URL

                try:
                    image_response = requests.get(image_url, stream=True)
                    image_response.raise_for_status()

                    # Load the image using OpenCV
                    image_array = np.asarray(bytearray(image_response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    # Detect faces using YOLOv8
                    results = model(image)
                    faces = []

                    # Filter detections for the 'face' class (assuming class 0 is 'face')
                    for result in results[0].boxes.data.tolist():
                        x_min, y_min, x_max, y_max, conf, cls = result
                        if int(cls) == 0:  # Assuming class 0 corresponds to 'face'
                            faces.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

                    if len(faces) > 0:
                        # Process only the first detected face
                        x, y, w, h = faces[0]
                        face = image[y:y+h, x:x+w]  # Crop the face

                        # Create a real face embedding using DeepFace
                        face_embedding = create_face_embedding(face)

                        if face_embedding:
                            face_embeddings.append({
                                "_id": person["_id"],
                                "name": person["name"],
                                "cmsId": person["cmsId"],
                                "status": person["status"],
                                "imageType": image_type,
                                "imageEmbedding": face_embedding
                            })
                        else:
                            print(f"Failed to generate embedding for {image_type} of {person['name']}")
                    else:
                        print(f"No face detected in {image_type} for {person['name']}")

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching image from URL {image_url}: {e}")
                except Exception as e:
                    print(f"Error processing image {image_type} for {person['name']}: {e}")

        # Save the face embeddings to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(face_embeddings, f)
        print(f"Face embeddings saved to {output_file}")

    except Exception as e:
        print(f"Error processing embeddings: {e}")

if __name__ == "__main__":
    fetch_and_store_embeddings()