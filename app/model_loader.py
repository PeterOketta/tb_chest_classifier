import os
import requests
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np
import logging
import sys

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

MODEL_PATH = "model/classification_model.keras"
CLASS_NAMES = ["Active TB", "Inactive or Healed TB", "Normal", "Others"]
MODEL_URL = "https://drive.google.com/uc?export=download&id=1dohEJ5qB1AM-M_jNbFPtRYKisEdts5hx"

def download_model():
    """Download the model from Google Drive if it's not present."""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        logger.info(f"Model directory status: {os.path.dirname(MODEL_PATH)} {'exists' if os.path.exists(os.path.dirname(MODEL_PATH)) else 'does not exist'}")
        
        # Check if model already exists and is not empty
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            logger.info(f"Model file already exists at {MODEL_PATH} with size {os.path.getsize(MODEL_PATH)} bytes")
            return
            
        logger.info("Starting model download from Google Drive...")
        
        # Set up session with headers to mimic browser
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First request to get the confirmation token
        response = session.get(MODEL_URL, headers=headers, stream=True)
        
        # Check for Google Drive warning page
        if 'Content-Disposition' not in response.headers:
            logger.info("Handling Google Drive download warning...")
            # Extract confirmation token
            for line in response.iter_lines():
                if b'confirm=' in line:
                    token = line.decode().split('confirm=')[1].split('&')[0]
                    logger.info(f"Found confirmation token: {token}")
                    response = session.get(f"{MODEL_URL}&confirm={token}", headers=headers, stream=True)
                    break
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model. Status code: {response.status_code}")
        
        # Download with progress tracking
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        logger.info(f"Starting file download, total size: {total_size} bytes")
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    downloaded += len(chunk)
                    f.write(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        if percent % 10 == 0:  # Log every 10%
                            logger.info(f"Download progress: {percent:.1f}%")
        
        # Verify downloaded file
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            logger.info(f"Download completed. File size: {file_size} bytes")
            if file_size == 0:
                raise RuntimeError("Downloaded file is empty")
            if total_size > 0 and file_size != total_size:
                raise RuntimeError(f"Downloaded file size ({file_size}) doesn't match expected size ({total_size})")
        else:
            raise RuntimeError("File was not created after download")
            
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}", exc_info=True)
        # Clean up partially downloaded file
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(f"Failed to download model: {str(e)}")

def load_model():
    """Load the model from the specified path."""
    try:
        # Ensure the model file is downloaded
        download_model()
        
        # Verify file exists and has content
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        file_size = os.path.getsize(MODEL_PATH)
        if file_size == 0:
            raise ValueError(f"Model file at {MODEL_PATH} is empty")
            
        logger.info(f"Attempting to load model from {MODEL_PATH} (size: {file_size} bytes)")
        model = keras_load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Basic model verification
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model doesn't have required 'predict' method")
            
        return model
        
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load model: {str(e)}")