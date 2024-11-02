from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import numpy as np
from monai.transforms import (
    LoadImage,
    Resize,
    ScaleIntensity
)
import requests
import tensorflow as tf
from app.model_loader import load_model, CLASS_NAMES
import tempfile
import os
import pydicom

app = FastAPI()

# Embedded credentials
DICOM_USERNAME = "admin"
DICOM_PASSWORD = "password123"

class ImageURL(BaseModel):
    url: HttpUrl

# Load the model
try:
    model = load_model()
except FileNotFoundError as e:
    raise RuntimeError(f"The Model could not be loaded: {e}")

# MONAI transforms for resizing and normalizing
resize_transform = Resize(spatial_size=(150, 150))
intensity_transform = ScaleIntensity(minv=0.0, maxv=1.0)

def preprocess_dicom_slice(dicom_data: bytes):
    """
    Load and preprocess only a single slice from a 3D DICOM for 2D model prediction.
    Ensures output is in RGB format (3 channels) as expected by the model.
    """
    try:
        # Save DICOM data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            dicom_file_path = tmp_file.name
            tmp_file.write(dicom_data)

        # Load the DICOM using pydicom to access metadata and slice directly
        dicom = pydicom.dcmread(dicom_file_path)

        # Extract the middle slice from the pixel array
        volume = dicom.pixel_array.astype(np.float32)
        depth_dim = np.argmin(volume.shape)
        middle_index = volume.shape[depth_dim] // 2

        # Slicing based on the depth dimension to get a single 2D slice
        slicer = [slice(None)] * volume.ndim
        slicer[depth_dim] = middle_index
        middle_slice = volume[tuple(slicer)]

        # Clean up temporary file
        os.remove(dicom_file_path)

        # Process the middle slice
        middle_slice = np.squeeze(middle_slice)  # Remove singleton dimensions
        middle_slice = middle_slice.astype(np.float32)

        # Add channel dimension for MONAI transforms
        middle_slice = middle_slice[None]

        # Resize the slice
        resized_slice = resize_transform(middle_slice)
        
        # Normalize the slice
        normalized_slice = intensity_transform(resized_slice)
        
        # Remove channel dimension after normalization
        normalized_slice = np.squeeze(normalized_slice)
        
        # Convert grayscale to RGB by stacking the same values three times
        rgb_slice = np.stack([normalized_slice] * 3, axis=-1)
        
        # Add batch dimension for model input
        final_slice = np.expand_dims(rgb_slice, axis=0)
        
        # Ensure the final shape is (1, 150, 150, 3)
        assert final_slice.shape == (1, 150, 150, 3), f"Unexpected shape: {final_slice.shape}"
        
        return final_slice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM file: {str(e)}")

@app.post("/predict/")
async def predict(url_data: ImageURL):
    """
    Predict from a single DICOM slice using embedded authentication and MONAI transforms.
    """
    try:
        # Fetch image from URL with authentication
        response = requests.get(
            url_data.url, 
            auth=(DICOM_USERNAME, DICOM_PASSWORD),
            timeout=10,
            verify=False
        )
        response.raise_for_status()
        dicom_data = response.content

        # Preprocess the DICOM data
        preprocessed_image = preprocess_dicom_slice(dicom_data)
        
        # Run prediction
        prediction = model.predict(preprocessed_image)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[0][predicted_index]
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "image_info": {
                "width": preprocessed_image.shape[1],
                "height": preprocessed_image.shape[2],
                "channels": preprocessed_image.shape[3]
            }
        }
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}