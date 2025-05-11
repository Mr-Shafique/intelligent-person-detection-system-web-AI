import pickle
import json

def print_face_embeddings(file_path):
    try:
        with open(file_path, "rb") as f:
            face_embeddings = pickle.load(f)

            for embedding in face_embeddings:
                print(f"ID: {embedding['_id']}")
                print(f"Name: {embedding['name']}")
                print(f"CMS ID: {embedding['cmsId']}")
                print(f"Status: {embedding['status']}")
                print(f"Image Type: {embedding['imageType']}")
                print(f"Image Embedding: {embedding['imageEmbedding'][:10]}... (truncated)")  # Print first 10 values for brevity
                print("-" * 40)
    except Exception as e:
        print(f"Error reading the pickle file: {e}")

if __name__ == "__main__":
    file_path = "face_embeddings.pkl"
    print_face_embeddings(file_path)
