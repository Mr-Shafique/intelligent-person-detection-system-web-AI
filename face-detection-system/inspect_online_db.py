import pickle
import os
import numpy as np

# Define the path to the embeddings file (relative to this script)
ONLINE_EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_face_embeddings.pkl")

print(f"Attempting to load data from: {os.path.abspath(ONLINE_EMBEDDINGS_FILE)}")

try:
    # Check if the file exists
    if not os.path.exists(ONLINE_EMBEDDINGS_FILE):
        print(f"Error: File not found at {ONLINE_EMBEDDINGS_FILE}")
    else:
        # Open the file in binary read mode
        with open(ONLINE_EMBEDDINGS_FILE, 'rb') as f:
            # Load the data from the pickle file
            online_database = pickle.load(f)

        # Check if the loaded data is a list
        if isinstance(online_database, list):
            print(f"\nSuccessfully loaded {len(online_database)} entries.")
            print("-" * 30)

            # Print details for each entry
            for i, entry in enumerate(online_database):
                print(f"Entry #{i + 1}:")
                name = entry.get('name', 'N/A')
                cmsId = entry.get('cmsId', 'N/A')
                status = entry.get('status', 'N/A')
                embedding = entry.get('embedding')
                source_image = entry.get('source_image', 'N/A')

                print(f"  Name: {name}")
                print(f"  CMS ID: {cmsId}")
                print(f"  Status: {status}")
                print(f"  Source Image: {source_image}")

                # Check if embedding exists and print its type and shape/length
                if embedding is not None:
                    if isinstance(embedding, (np.ndarray, list)):
                        print(f"  Embedding Type: {type(embedding)}")
                        try:
                            print(f"  Embedding Shape/Length: {np.shape(embedding)}")
                        except:
                             print(f"  Embedding Length: {len(embedding)}")
                    else:
                        print(f"  Embedding Type: {type(embedding)} (Unexpected)")
                else:
                    print("  Embedding: None")
                print("-" * 10)

        else:
            print(f"Error: Expected data to be a list, but got {type(online_database)}")

except FileNotFoundError:
    print(f"Error: File not found at {ONLINE_EMBEDDINGS_FILE}")
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle data. The file might be corrupted or not a valid pickle file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

