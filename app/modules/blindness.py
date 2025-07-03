from fastapi import APIRouter, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import os
from app.utils.preprocessing import preprocess_image

router = APIRouter()

# Absolute path to weights file
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(current_dir, '..', 'models', 'blindness_model.h5'))

def create_blindness_model():
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Build model and load weights
model = create_blindness_model()
model.load_weights(MODEL_PATH)

@router.post("/predict")
async def predict_blindness(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents, target_size=(224, 224))
    prediction = model.predict(processed_image)
    predicted_class = int(np.argmax(prediction[0]))
    return {
        "prediction": predicted_class
    }
