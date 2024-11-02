from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import numpy as np
from PIL import Image
from fastapi import File, UploadFile
from monai.transforms import Resize, ScaleIntensity
import requests
import tempfile
import os
import pydicom
from app.model_loader import load_model, CLASS_NAMES, preprocess_image

app = FastAPI()

# Embedded credentials
DICOM_USERNAME = "admin"
DICOM_PASSWORD = "password123"

# Initialize the model variable as None
model = None

# MONAI transforms for resizing and normalizing
resize_transform = Resize(spatial_size=(150, 150))
intensity_transform = ScaleIntensity(minv=0.0, maxv=1.0)

class ImageURL(BaseModel):
    url: HttpUrl

def get_model():
    global model
    if model is None:
        print("Loading the model...")
        model = load_model()
    return model

def preprocess_dicom_slice(dicom_data: bytes):
    """
    Load and preprocess only a single slice from a 3D DICOM for 2D model prediction.
    Ensures output is in RGB format (3 channels) as expected by the model.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            dicom_file_path = tmp_file.name
            tmp_file.write(dicom_data)

        dicom = pydicom.dcmread(dicom_file_path)
        volume = dicom.pixel_array.astype(np.float32)
        depth_dim = np.argmin(volume.shape)
        middle_index = volume.shape[depth_dim] // 2

        slicer = [slice(None)] * volume.ndim
        slicer[depth_dim] = middle_index
        middle_slice = volume[tuple(slicer)]

        os.remove(dicom_file_path)
        
        middle_slice = np.squeeze(middle_slice).astype(np.float32)
        middle_slice = middle_slice[None]

        resized_slice = resize_transform(middle_slice)
        normalized_slice = intensity_transform(resized_slice)
        normalized_slice = np.squeeze(normalized_slice)
        rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
        final_slice = np.expand_dims(rgb_slice, axis=0)
        
        assert final_slice.shape == (1, 150, 150, 3), f"Unexpected shape: {final_slice.shape}"
        
        return final_slice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM file: {str(e)}")

@app.post("/predict-dicom/")
async def predict(url_data: ImageURL):
    try:
        response = requests.get(
            url_data.url, 
            auth=(DICOM_USERNAME, DICOM_PASSWORD),
            timeout=10,
            verify=False
        )
        response.raise_for_status()
        dicom_data = response.content

        preprocessed_image = preprocess_dicom_slice(dicom_data)
        
        # Get the model instance
        model_instance = get_model()
        prediction = model_instance.predict(preprocessed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[0][predicted_index]
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format. Only PNG, JPEG, and JPG are allowed.")

        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
        
        # Get the model instance
        model_instance = get_model()
        prediction = model_instance.predict(preprocessed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[0][predicted_index]
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image file: {str(e)}")
        
@app.get("/health")
async def health_check():
    model_loaded = model is not None
    return {"status": "healthy", "model_loaded": model_loaded}
