import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import numpy as np
import os


MODEL_PATH = "/app/model/classification_model.keras"

CLASS_NAMES = [" Active TB", "Inactive or Healed TB", "Normal","Others"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")
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
