import os
import requests
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np

# Path to the model file within the Docker container
MODEL_PATH = "model/classification_model.keras"
CLASS_NAMES = ["Active TB", "Inactive or Healed TB", "Normal", "Others"]

# Pre-authenticated URL to download the model
MODEL_URL = "https://storage.cloud.google.com/classification_model_tb/classification_model.keras"

def download_model():
    """Download the model from a pre-authenticated URL if it's not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
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
