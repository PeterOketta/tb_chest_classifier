import os
import requests
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np

# Path to the model file within the Docker container
MODEL_PATH = "model/classification_model.keras"
CLASS_NAMES = ["Active TB", "Inactive or Healed TB", "Normal", "Others"]

# Google Drive direct download link with your file's ID
MODEL_URL = "https://drive.google.com/uc?export=download&id=1dohEJ5qB1AM-M_jNbFPtRYKisEdts5hx"

def download_model():
    """Download the model from Google Drive if it's not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from Google Drive...")
        
        # Streaming the download to handle large files
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise RuntimeError(f"Failed to download model: {response.status_code}")

def load_model():
    # Ensure the model file is downloaded
    download_model()
    
    try:
        model = keras_load_model(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def preprocess_image(image: Image.Image):
    img_width, img_height = 150, 150
    image = image.resize((img_width, img_height))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
