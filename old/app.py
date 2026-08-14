"""
Model utilities for DFU classification
(Streamlit version removed - use testa.py for Flask app)
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import efficientnet

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    IMG_SIZE = (300, 300)
    CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
    MODEL_PATH = 'best_model_stage2.h5'

cfg = Config()

# ==========================================
# MODEL LOADING
# ==========================================
def load_tf_model():
    """Load TensorFlow model (no caching - use manually in Flask)"""
    try:
        model = tf.keras.models.load_model(cfg.MODEL_PATH, compile=False)
        print(f"✅ Model loaded: {cfg.MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found: {cfg.MODEL_PATH}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess_image(image):
    """
    Preprocess image for EfficientNetB3:
    1. Convert to RGB
    2. Resize to (300, 300)
    3. Convert to array
    4. Apply EfficientNet preprocessing
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(cfg.IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = efficientnet.preprocess_input(img_array)
    
    return img_array

def predict(model, image):
    """Run prediction on preprocessed image"""
    if model is None:
        return None, 0.0
    
    preds = model.predict(image, verbose=0)
    probs = preds[0]
    pred_idx = np.argmax(probs)
    pred_class = cfg.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    return pred_class, confidence