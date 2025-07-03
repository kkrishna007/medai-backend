from PIL import Image
import numpy as np
import io
import cv2

def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def get_original_image(image_bytes, target_size=(224, 224)):

    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    return np.array(image)

def preprocess_brain_image(image_bytes, target_size=(150, 150)):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed. File may be corrupted or not an image.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)  # Shape: (1, 150, 150, 3)
    except Exception as e:
        # Add this error logging
        print(f"❌ BRAIN TUMOR PREPROCESSING ERROR: {str(e)}")
        raise



def preprocess_batch(images, target_size=(224, 224)):
    if len(images.shape) == 3:  # Single image
        images = np.expand_dims(images, axis=0)
        
    # Ensure images are float32 and normalized
    if images.dtype != np.float32:
        images = images.astype(np.float32)
        
    if images.max() > 1.0:
        images = images / 255.0
        
    return images
