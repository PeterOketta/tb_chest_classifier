# app/model_loader.py

import os
import requests
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "model/classification_model.keras"
CLASS_NAMES = ["Active TB", "Inactive or Healed TB", "Normal", "Others"]
MODEL_URL = "https://drive.google.com/uc?export=download&id=1dohEJ5qB1AM-M_jNbFPtRYKisEdts5hx"

def download_model():
    """Download the model from Google Drive if it's not present."""
    try:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            logger.info(f"Model directory created at {os.path.dirname(MODEL_PATH)}")
            logger.info("Downloading model from Google Drive...")
            
            # First request to get the confirmation token if needed
            session = requests.Session()
            response = session.get(MODEL_URL, stream=True)
            
            # Check if there's a download warning (file size > 100MB)
            if "download_warning" in response.text:
                logger.info("Large file detected, handling Google Drive warning...")
                token = response.text.split('"')[1]
                response = session.get(f"{MODEL_URL}&confirm={token}", stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            downloaded += len(chunk)
                            f.write(chunk)
                            if total_size:
                                percent = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {percent:.1f}%")
                
                logger.info(f"Model downloaded successfully to {MODEL_PATH}")
                logger.info(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
            else:
                raise RuntimeError(f"Failed to download model. Status code: {response.status_code}")
        else:
            logger.info(f"Model file already exists at {MODEL_PATH}")
            
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}")
        raise RuntimeError(f"Failed to download model: {str(e)}")

def load_model():
    """Load the model from the specified path."""
    try:
        # Ensure the model file is downloaded
        download_model()
        
        # Check if the file exists and has content
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        if os.path.getsize(MODEL_PATH) == 0:
            raise ValueError(f"Model file at {MODEL_PATH} is empty")
        
        logger.info("Loading model...")
        model = keras_load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def preprocess_image(image: Image.Image):
    """Preprocess the input image for model prediction."""
    try:
        img_width, img_height = 150, 150
        image = image.resize((img_width, img_height))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise RuntimeError(f"Failed to preprocess image: {str(e)}")