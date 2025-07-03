from fastapi import APIRouter, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import os
from app.utils.preprocessing import preprocess_image

router = APIRouter()

# Correct absolute path calculation for model file
current_dir = os.path.dirname(os.path.abspath(__file__))  # This works in .py files
MODEL_PATH = os.path.normpath(os.path.join(current_dir, '..', 'models', 'pneumonia_model.h5'))

# Optional: print for debugging
#print(f"Loading pneumonia model from: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

@router.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents, target_size=(224, 224))
    prediction = model.predict(processed_image)
    predicted_class = int(prediction[0][0] > 0.5)
    return {
        "prediction": predicted_class,
        "result": "Pneumonia" if predicted_class else "Normal"
    }
