# main application file for diabetic foot ulcer analysis web app
import base64
import io
import os
from datetime import datetime  # <--- NEW IMPORT
from dotenv import load_dotenv
import cv2
import numpy as np
import tensorflow as tf
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications import efficientnet
from lime import lime_image
from skimage.segmentation import mark_boundaries
from groq import Groq

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)

# --- API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- PATHS (Updated) ---
TF_MODEL_PATH = "final_model.h5"
TORCH_SEG_PATH = "new_unet_seg.pth"

# --- SETTINGS ---
CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
IMG_SIZE = (300, 300)      # Classification Input
SEG_SIZE = (224, 224)      # Segmentation Input
PIXELS_PER_CM = 38.0       # Calibration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. LOAD MODELS
# ==========================================
print(f"⚙️ System Device: {DEVICE}")

# A. TensorFlow Classifier
print("⏳ Loading Classifier...")
tf_model = None
try:
    tf_model = tf.keras.models.load_model(TF_MODEL_PATH, compile=False)
    print("✅ Classifier Loaded.")
except Exception as e:
    print(f"❌ Classifier Error: {e}")

# B. PyTorch Segmentation
print("⏳ Loading Segmentation...")
seg_model = None
try:
    seg_model = smp.Unet(
        encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1, activation=None
    ).to(DEVICE)
    
    if os.path.isfile(TORCH_SEG_PATH):
        seg_model.load_state_dict(torch.load(TORCH_SEG_PATH, map_location=DEVICE))
        seg_model.eval()
        print("✅ Segmentation Loaded.")
    else:
        print(f"⚠️ Segmentation weights not found at {TORCH_SEG_PATH}")
except Exception as e:
    print(f"❌ Segmentation Error: {e}")

# C. MiDaS Depth Estimation
print("⏳ Loading MiDaS Depth Model...")
midas_model = None
midas_transform = None
try:
    model_type = "MiDaS_small" 
    midas_model = torch.hub.load("intel-isl/MiDaS", model_type).to(DEVICE)
    midas_model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.small_transform
    print(f"✅ MiDaS ({model_type}) Loaded.")
except Exception as e:
    print(f"❌ MiDaS Error: {e}")

# D. Explainers (GradCAM & LIME)
grad_model = None
if tf_model:
    target_layer = None
    for layer in tf_model.layers:
        if 'efficientnet' in layer.name.lower():
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    target_layer = sub.name
                    grad_model = tf.keras.models.Model([layer.input], [layer.get_layer(sub.name).output, layer.output])
                    break
    if not grad_model: 
         for layer in reversed(tf_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                grad_model = tf.keras.models.Model([tf_model.inputs], [tf_model.get_layer(layer.name).output, tf_model.output])
                break

lime_explainer = lime_image.LimeImageExplainer()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def array_to_base64(img_array):
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def get_depth_map(img_rgb):
    if midas_model is None: 
        return np.zeros(img_rgb.shape[:2]), 0.0
    input_batch = midas_transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_score = np.mean(depth_norm)
    return depth_norm, round(float(depth_score), 2)

def predict_lime_fn(images):
    batch = efficientnet.preprocess_input(np.array(images))
    return tf_model.predict(batch, verbose=0)

def generate_llm_report(symptoms, ai_results):
    if not GROQ_API_KEY:
        return "LLM Analysis Unavailable: GROQ_API_KEY not set."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        symptom_text = "\n".join([f"- {k}: {v}" for k, v in symptoms.items()])
        ai_text = "\n".join([f"- {k}: {v}" for k, v in ai_results.items()])
        prompt = f"""
        You are an expert medical assistant specializing in Diabetic Foot Ulcers.
        Analyze this patient case based on their reported symptoms and AI Computer Vision findings.
        PATIENT SYMPTOMS: {symptom_text}
        AI IMAGE ANALYSIS: {ai_text}
        OUTPUT FORMAT:
        1. **Summary**: Synthesis of visual and symptom data.
        2. **Risk Assessment**: High/Medium/Low urgency.
        3. **Recommendations**: 3-4 actionable steps.
        4. **Disclaimer**: State that you are an AI.
        Keep it concise (max 200 words) and empathetic.
        """
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    
    if request.method == 'POST':
        symptoms = {
            "Redness": request.form.get('redness', 'No'),
            "Swelling": request.form.get('swelling', 'No'),
            "Odor": request.form.get('odor', 'No'),
            "Pain": request.form.get('pain', 'No'),
            "Discharge": request.form.get('discharge', 'No'),
            "Fever": request.form.get('fever', 'No')
        }
        
        # Add current date for the report
        context['report_date'] = datetime.now().strftime("%d %B, %Y at %H:%M")
        context['symptoms'] = symptoms # Pass back for display in PDF
        
        file = request.files.get('file')
        
        if not file or file.filename == '':
            context['error'] = "Please upload an image."
            return render_template('indexb.html', **context)
        
        if not tf_model:
            context['error'] = "Classifier model not loaded."
            return render_template('indexb.html', **context)
    
        try:
            img_pil = Image.open(file.stream).convert('RGB')
            img_np = np.array(img_pil)
            img_tf = img_pil.resize(IMG_SIZE)
            img_tf_arr = tf.keras.preprocessing.image.img_to_array(img_tf)
            img_tf_batch = np.expand_dims(img_tf_arr, axis=0)
            img_pre = efficientnet.preprocess_input(img_tf_batch.copy())

            # A. CLASSIFICATION
            preds = tf_model.predict(img_pre, verbose=0)
            pred_idx = np.argmax(preds[0])
            pred_label = CLASS_NAMES[pred_idx]
            confidence = round(float(preds[0][pred_idx]) * 100, 2)
            
            context['prediction'] = pred_label
            context['confidence'] = confidence
            context['img_data'] = array_to_base64(img_np)
            
            # B. GRAD-CAM
            if grad_model:
                with tf.GradientTape() as tape:
                    conv_out, pred_out = grad_model(img_pre)
                    loss = pred_out[:, pred_idx]
                grads = tape.gradient(loss, conv_out)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis])
                heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
                heatmap_resized = cv2.resize(heatmap.numpy(), (img_np.shape[1], img_np.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
                context['gradcam_data'] = array_to_base64(overlay)

            # C. SEGMENTATION
            area_cm2 = 0; width_cm = 0
            if seg_model:
                seg_t = A.Compose([A.Resize(SEG_SIZE[0], SEG_SIZE[1]), A.Normalize(), ToTensorV2()])
                input_t = seg_t(image=img_np)['image'].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    mask = (torch.sigmoid(seg_model(input_t)) > 0.5).float().squeeze().cpu().numpy()
                mask_uint8 = cv2.resize((mask*255).astype(np.uint8), (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                context['seg_data'] = array_to_base64(mask_uint8)
                area_pixels = np.count_nonzero(mask_uint8)
                area_cm2 = round(area_pixels / (PIXELS_PER_CM**2), 2)
                x, y, w, h = cv2.boundingRect(mask_uint8)
                width_cm = round(w / PIXELS_PER_CM, 2)

            # D. DEPTH
            depth_map, depth_score = get_depth_map(img_np)
            depth_colored = cv2.applyColorMap(np.uint8(255 * depth_map), cv2.COLORMAP_INFERNO)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            context['depth_data'] = array_to_base64(depth_colored)

            # E. LIME
            try:
                lime_exp = lime_explainer.explain_instance(np.array(img_tf), predict_lime_fn, top_labels=1, hide_color=0, num_samples=100)
                temp_lime, mask_lime = lime_exp.get_image_and_mask(lime_exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
                lime_boundary = mark_boundaries(temp_lime/255.0, mask_lime, color=(1, 1, 0))
                context['lime_data'] = array_to_base64((lime_boundary * 255).astype(np.uint8))
            except: context['lime_data'] = None
            
            # F. REPORT
            ai_results = {
                "Classification": pred_label, "Confidence": f"{confidence}%",
                "Wound Area": f"{area_cm2} cm2", "Max Width": f"{width_cm} cm",
                "Relative Depth Index": depth_score
            }
            context['metrics'] = ai_results 
            context['llm_report'] = generate_llm_report(symptoms, ai_results)

        except Exception as e:
            context['error'] = f"Analysis Failed: {str(e)}"
            import traceback; traceback.print_exc()

    return render_template('indexb.html', **context)

if __name__ == '__main__':
    app.run(debug=True, port=5000)