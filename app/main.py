from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from app.model_loader import load_model, preprocess_image, CLASS_NAMES

app = FastAPI()

try:
    model = load_model()
except FileNotFoundError as e:
    raise RuntimeError(f"The Model could not be loaded: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png","image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Use JPEG or PNG.")
        
        # Open and preprocess the image
        image = Image.open(file.file)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_index]
        
        return {"prediction": predicted_class}
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
