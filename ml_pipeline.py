"""
Complete ML Pipeline for DFU Classification
Includes: Classification, GradCAM, Segmentation, Depth, LIME, LLM Report
Loaded ONCE at startup, shared across threads
"""
import numpy as np
import tensorflow as tf
import torch
import cv2
import io
import os
import base64
from PIL import Image
from tensorflow.keras.applications import efficientnet
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths
    #TF_MODEL_PATH = 'final_model.h5'
    TF_MODEL_PATH = 'final_dfu_model_weighted.h5'  # Updated path for the classification model
    TORCH_SEG_PATH = 'new_unet_seg.pth'
    
    # Model settings
    CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
    IMG_SIZE = (300, 300)      # Classification input
    SEG_SIZE = (224, 224)      # Segmentation input
    PIXELS_PER_CM = 38.0       # Calibration for wound metrics
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# ==========================================
# GLOBAL MODEL INSTANCES (loaded once, shared across threads)
# ==========================================
_tf_model = None
_seg_model = None
_midas_model = None
_midas_transform = None
_grad_model = None
_lime_explainer = None

def initialize_models():
    """
    Load all ML models at startup (ONCE).
    Called in Flask app initialization.
    Models are shared across all threads.
    """
    global _tf_model, _seg_model, _midas_model, _midas_transform, _grad_model, _lime_explainer
    
    print("\n" + "=" * 60)
    print(" INITIALIZING ML MODELS")
    print("=" * 60)
    print(f"  Device: {cfg.DEVICE}\n")
    
    # A. TensorFlow Classification Model
    print(" Loading TensorFlow Classifier...")
    print(f"   Looking for: {cfg.TF_MODEL_PATH}")
    print(f"   Absolute path: {os.path.abspath(cfg.TF_MODEL_PATH)}")
    print(f"   File exists: {os.path.isfile(cfg.TF_MODEL_PATH)}\n")
    
    try:
        if not os.path.isfile(cfg.TF_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {os.path.abspath(cfg.TF_MODEL_PATH)}")
        
        _tf_model = tf.keras.models.load_model(cfg.TF_MODEL_PATH, compile=False)
        print(" Classification model loaded\n")
    except FileNotFoundError as e:
        print(f" Model file missing: {e}\n")
        print("   Available files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                print(f"     - {f}")
        print()
        _tf_model = None
    except Exception as e:
        print(f" Error loading classifier: {e}\n")
        _tf_model = None
    
    # B. PyTorch Segmentation Model
    print(" Loading PyTorch Segmentation Model...")
    try:
        _seg_model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        ).to(cfg.DEVICE)
        
        if os.path.isfile(cfg.TORCH_SEG_PATH):
            _seg_model.load_state_dict(torch.load(cfg.TORCH_SEG_PATH, map_location=cfg.DEVICE))
            _seg_model.eval()
            print(" Segmentation model loaded\n")
        else:
            print(f"  Segmentation weights not found: {cfg.TORCH_SEG_PATH}\n")
            _seg_model = None
    except Exception as e:
        print(f" Error loading segmentation: {e}\n")
        _seg_model = None
    
    # C. MiDaS Depth Estimation Model
    print(" Loading MiDaS Depth Model...")
    try:
        model_type = "MiDaS_small"
        _midas_model = torch.hub.load("intel-isl/MiDaS", model_type).to(cfg.DEVICE)
        _midas_model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _midas_transform = midas_transforms.small_transform
        print(f" MiDaS depth model loaded ({model_type})\n")
    except Exception as e:
        print(f"  MiDaS not available: {e}\n")
        _midas_model = None
        _midas_transform = None
    
    # D. GradCAM Setup
    print(" Setting up GradCAM...")
    if _tf_model:
        try:
            # Find EfficientNet backbone layer
            target_layer = None
            for layer in _tf_model.layers:
                if 'efficientnet' in layer.name.lower():
                    for sub in reversed(layer.layers):
                        if isinstance(sub, tf.keras.layers.Conv2D):
                            _grad_model = tf.keras.models.Model(
                                [layer.input],
                                [layer.get_layer(sub.name).output, layer.output]
                            )
                            target_layer = sub.name
                            break
            
            if not _grad_model:
                for layer in reversed(_tf_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        _grad_model = tf.keras.models.Model(
                            [_tf_model.inputs],
                            [_tf_model.get_layer(layer.name).output, _tf_model.output]
                        )
                        break
            
            if _grad_model:
                print(" GradCAM model ready\n")
            else:
                print("  Could not setup GradCAM\n")
        except Exception as e:
            print(f"  GradCAM setup failed: {e}\n")
    
    # E. LIME Explainer
    print("⏳ Initializing LIME Explainer...")
    try:
        _lime_explainer = lime_image.LimeImageExplainer()
        print(" LIME explainer ready\n")
    except Exception as e:
        print(f"  LIME initialization failed: {e}\n")
    
    print("=" * 60)
    print(" MODEL INITIALIZATION COMPLETE\n")

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def load_and_preprocess_image(file_path):
    """
    Load image from file and return both numpy array and PIL image
    
    Returns:
        Tuple: (tf_preprocessed_array, numpy_array, pil_image)
    """
    try:
        # Load image
        img_pil = Image.open(file_path).convert('RGB')
        img_np = np.array(img_pil)
        
        # For TensorFlow classification
        img_tf = img_pil.resize(cfg.IMG_SIZE)
        img_tf_arr = tf.keras.preprocessing.image.img_to_array(img_tf)
        img_tf_batch = np.expand_dims(img_tf_arr, axis=0)
        img_pre = efficientnet.preprocess_input(img_tf_batch.copy())
        
        return img_pre, img_np, img_pil
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

# ==========================================
# CLASSIFICATION
# ==========================================

def classify_image(img_array):
    """
    Run EfficientNetB3 classification
    
    Args:
        img_array: Preprocessed image from load_and_preprocess_image()
    
    Returns:
        Tuple: (prediction_class, confidence_percent, probabilities_dict)
    """
    if _tf_model is None:
        raise Exception("Classification model not loaded")
    
    try:
        preds = _tf_model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = cfg.CLASS_NAMES[pred_idx]
        confidence = round(float(preds[0][pred_idx]) * 100, 2)
        
        # Create probability dict
        prob_dict = {cfg.CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(cfg.CLASS_NAMES))}
        
        return pred_class, confidence, prob_dict
    
    except Exception as e:
        raise Exception(f"Classification error: {e}")

# ==========================================
# XAI - GRADCAM
# ==========================================

def generate_gradcam(img_array, img_np, pred_idx):
    """
    Generate GradCAM heatmap overlay
    
    Args:
        img_array: Preprocessed TensorFlow array
        img_np: Original numpy image
        pred_idx: Predicted class index
    
    Returns:
        Base64 encoded PNG or None
    """
    if _grad_model is None:
        return None
    
    try:
        with tf.GradientTape() as tape:
            conv_out, pred_out = _grad_model(img_array)
            loss = pred_out[:, pred_idx]
        
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        heatmap_resized = cv2.resize(heatmap.numpy(), (img_np.shape[1], img_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
        
        return array_to_base64(overlay)
    
    except Exception as e:
        print(f"  GradCAM generation failed: {e}")
        return None

# ==========================================
# XAI - LIME
# ==========================================

def generate_lime(img_pil, pred_idx):
    """
    Generate LIME explanation with boundary overlay
    
    Args:
        img_pil: Original PIL image
        pred_idx: Predicted class index
    
    Returns:
        Base64 encoded PNG or None
    """
    if _lime_explainer is None or _tf_model is None:
        return None
    
    try:
        # Resize for LIME (LIME uses smaller images)
        img_tf = img_pil.resize(cfg.IMG_SIZE)
        
        def predict_lime_fn(images):
            batch = efficientnet.preprocess_input(np.array(images))
            return _tf_model.predict(batch, verbose=0)
        
        # Generate LIME explanation
        lime_exp = _lime_explainer.explain_instance(
            np.array(img_tf),
            predict_lime_fn,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )
        
        # Get top prediction
        temp_lime, mask_lime = lime_exp.get_image_and_mask(
            pred_idx,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        # Add boundaries
        lime_boundary = mark_boundaries(temp_lime / 255.0, mask_lime, color=(1, 1, 0))
        
        return array_to_base64((lime_boundary * 255).astype(np.uint8))
    
    except Exception as e:
        print(f"  LIME generation failed: {e}")
        return None

# ==========================================
# SEGMENTATION
# ==========================================

def segment_wound(img_np):
    """
    Segment wound region using PyTorch U-Net
    
    Args:
        img_np: Original numpy image
    
    Returns:
        Tuple: (segmentation_mask_base64, metrics_dict)
    """
    if _seg_model is None:
        return None, {
            'Wound Area': '0 cm²',
            'Max Width': '0 cm',
            'Relative Depth Index': '0.0'
        }
    
    try:
        # Prepare image for segmentation
        seg_t = A.Compose([
            A.Resize(cfg.SEG_SIZE[0], cfg.SEG_SIZE[1]),
            A.Normalize(),
            ToTensorV2()
        ])
        
        input_t = seg_t(image=img_np)['image'].unsqueeze(0).to(cfg.DEVICE)
        
        # Run segmentation
        with torch.no_grad():
            mask = (torch.sigmoid(_seg_model(input_t)) > 0.5).float().squeeze().cpu().numpy()
        
        # Resize mask back to original size
        mask_uint8 = cv2.resize(
            (mask * 255).astype(np.uint8),
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Calculate metrics
        area_pixels = np.count_nonzero(mask_uint8)
        area_cm2 = round(area_pixels / (cfg.PIXELS_PER_CM ** 2), 2)
        
        x, y, w, h = cv2.boundingRect(mask_uint8)
        width_cm = round(w / cfg.PIXELS_PER_CM, 2)
        
        metrics = {
            'Wound Area': f'{area_cm2} cm²',
            'Max Width': f'{width_cm} cm',
            'Relative Depth Index': '0.0'  # Will be updated by depth
        }
        
        return array_to_base64(mask_uint8), metrics
    
    except Exception as e:
        print(f"  Segmentation failed: {e}")
        return None, {}

# ==========================================
# DEPTH ESTIMATION
# ==========================================

def estimate_depth(img_np):
    """
    Estimate depth map using MiDaS model
    
    Args:
        img_np: Original numpy image
    
    Returns:
        Tuple: (depth_map_base64, depth_score)
    """
    if _midas_model is None or _midas_transform is None:
        return None, 0.0
    
    try:
        # Convert BGR to RGB if needed
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.max() > 1 else img_np
        else:
            img_rgb = img_np
        
        # Prepare for MiDaS
        input_batch = _midas_transform(img_rgb).to(cfg.DEVICE)
        
        # Run depth estimation
        with torch.no_grad():
            prediction = _midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Normalize depth map
        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        depth_score = round(float(np.mean(depth_norm)), 2)
        
        # Create colored depth visualization
        depth_colored = cv2.applyColorMap(np.uint8(255 * depth_norm), cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        return array_to_base64(depth_colored), depth_score
    
    except Exception as e:
        print(f"  Depth estimation failed: {e}")
        return None, 0.0

# ==========================================
# LLM REPORT GENERATION
# ==========================================

def generate_llm_report(symptoms_dict, ai_results):
    """
    Generate medical report using Groq LLM
    
    Args:
        symptoms_dict: Dictionary of reported symptoms
        ai_results: Dictionary of AI analysis results
    
    Returns:
        Medical report string
    """
    if not cfg.GROQ_API_KEY:
        return "LLM Analysis Unavailable: GROQ_API_KEY not set in environment."
    
    try:
        client = Groq(api_key=cfg.GROQ_API_KEY)
        
        # Format symptoms
        symptom_text = "\n".join([f"- {k}: {'Yes' if v else 'No'}" for k, v in symptoms_dict.items()])
        
        # Format AI results
        ai_text = "\n".join([f"- {k}: {v}" for k, v in ai_results.items()])
        
        # Create prompt
        prompt = f"""
You are an expert medical assistant specializing in Diabetic Foot Ulcers (DFU).
Analyze this patient case based on their reported symptoms and AI Computer Vision findings.

PATIENT REPORTED SYMPTOMS:
{symptom_text}

AI IMAGE ANALYSIS RESULTS:
{ai_text}

Please provide a comprehensive assessment in the following format:

1. **Clinical Summary**: Brief synthesis of visual and symptom data.
2. **Risk Assessment**: Classify as High/Medium/Low urgency.
3. **Key Findings**: Main observations from the image analysis.
4. **Recommendations**: 3-4 actionable clinical steps.
5. **Follow-up**: When to seek medical attention.

Keep the response concise (max 300 words) and use clear, professional language.
IMPORTANT: Include a disclaimer that this is AI-generated and requires professional medical review.
"""
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"LLM Report Generation Error: {str(e)}\nPlease check your Groq API key and internet connection."

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def array_to_base64(img_array):
    """Convert numpy array to base64 PNG string"""
    try:
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        img_pil = Image.fromarray(img_array)
        buff = io.BytesIO()
        img_pil.save(buff, format='PNG')
        return base64.b64encode(buff.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error converting array to base64: {e}")
        return ""

def image_to_base64(image):
    """Convert PIL image to base64 PNG string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64
    
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""
