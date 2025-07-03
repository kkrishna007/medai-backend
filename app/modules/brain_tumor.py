from fastapi import APIRouter, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import os
import logging  # Add this
import traceback  # Already added
from app.utils.preprocessing import preprocess_brain_image

router = APIRouter()

# Setup logger for this module
logger = logging.getLogger("brain_tumor_module")  # Add this

# Calculate the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(current_dir, '..', 'models', 'brain_tumor.h5')
MODEL_PATH = os.path.normpath(MODEL_PATH)  

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

@router.post("/predict")
async def predict_brain_tumor(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        logger.info(f"Received image bytes size: {len(contents)}")
        
        # Preprocess image
        processed_image = preprocess_brain_image(contents, target_size=(150, 150))
        
        # Get prediction
        prediction = model.predict(processed_image, verbose=1)
        predicted_class = int(np.argmax(prediction[0]))
        
        # Map prediction to tumor type
        tumor_map = {
            0: "Glioma Tumor", 1: "Meningioma Tumor",
            2: "No Tumor Found", 3: "Pituitary Tumor"
        }
        
        # Create result object
        result = {
            "prediction": predicted_class,
            "tumor_type": tumor_map[predicted_class],
            "confidence": float(prediction[0][predicted_class]),
            "raw_prediction": prediction[0].tolist()
        }
        
        logger.info(f"Returning to frontend: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Brain tumor prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Image processing error")
