"""
Pure job functions for RQ worker.
No Flask or Celery dependencies - this allows the worker to load models once and reuse them.
"""
import base64
import io
import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from tensorflow.keras.applications import efficientnet
from lime import lime_image
from skimage.segmentation import mark_boundaries
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
CLASS_NAMES = ['Both', 'Infection', 'Ischaemia', 'None']
IMG_SIZE = (300, 300)
SEG_SIZE = (224, 224)
PIXELS_PER_CM = 38.0
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Models will be loaded by the worker, not here
# This keeps tasks.py lightweight

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def array_to_base64(img_array):
    """Convert numpy array to base64 string for HTML display"""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def generate_llm_report(symptoms, ai_results):
    """Calls Groq API to generate report"""
    if not GROQ_API_KEY:
        return "LLM Analysis Unavailable: GROQ_API_KEY not set in environment."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        symptom_text = "\n".join([f"- {k}: {v}" for k, v in symptoms.items()])
        ai_text = "\n".join([f"- {k}: {v}" for k, v in ai_results.items()])
        
        prompt = f"""
        You are an expert medical assistant specializing in Diabetic Foot Ulcers.
        Analyze this patient case based on their reported symptoms and AI Computer Vision findings.
        
        PATIENT SYMPTOMS:
        {symptom_text}
        
        AI IMAGE ANALYSIS:
        {ai_text}
        
        OUTPUT FORMAT:
        1. **Summary**: Synthesis of visual and symptom data.
        2. **Risk Assessment**: High/Medium/Low urgency based on signs like "Infection", "Redness", "Necrotic" tissue.
        3. **Recommendations**: 3-4 actionable steps for the patient.
        4. **Disclaimer**: State that you are an AI and this is not a diagnosis.
        
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
# MAIN JOB FUNCTION
# ==========================================
def process_image_job(img_base64, symptoms, tf_model, seg_model, grad_model, midas_model, 
                      midas_transform, lime_explainer, device):
    """
    Main image processing job.
    
    Args:
        img_base64: Base64 encoded image
        symptoms: Dict of symptom data
        tf_model, seg_model, grad_model, midas_model, midas_transform, lime_explainer, device:
            Loaded models passed from worker (shared across jobs)
    
    Returns:
        Dict with prediction, confidence, images, metrics, report
    """
    # Decode image
    img_data = base64.b64decode(img_base64)
    img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
    img_np = np.array(img_pil)
    
    result = {}
    
    # --- A. CLASSIFICATION ---
    img_tf = img_pil.resize(IMG_SIZE)
    img_tf_arr = tf.keras.preprocessing.image.img_to_array(img_tf)
    img_tf_batch = np.expand_dims(img_tf_arr, axis=0)
    img_pre = efficientnet.preprocess_input(img_tf_batch.copy())
    
    preds = tf_model.predict(img_pre, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_label = CLASS_NAMES[pred_idx]
    confidence = round(float(preds[0][pred_idx]) * 100, 2)
    
    result['prediction'] = pred_label
    result['confidence'] = confidence
    result['img_data'] = array_to_base64(img_np)
    
    # --- B. GRAD-CAM ---
    if grad_model:
        try:
            with tf.GradientTape() as tape:
                conv_out, pred_out = grad_model(img_pre)
                loss = pred_out[:, pred_idx]
            grads = tape.gradient(loss, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis])
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            heatmap = heatmap.numpy()
            
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
            result['gradcam_data'] = array_to_base64(overlay)
        except Exception as e:
            print(f"GradCAM Error: {e}")
            result['gradcam_data'] = None
    
    # --- C. SEGMENTATION ---
    area_cm2 = 0
    width_cm = 0
    if seg_model:
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            seg_t = A.Compose([A.Resize(SEG_SIZE[0], SEG_SIZE[1]), A.Normalize(), ToTensorV2()])
            input_t = seg_t(image=img_np)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                mask = (torch.sigmoid(seg_model(input_t)) > 0.5).float().squeeze().cpu().numpy()
            
            mask_uint8 = cv2.resize((mask*255).astype(np.uint8), (img_np.shape[1], img_np.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            result['seg_data'] = array_to_base64(mask_uint8)
            
            area_pixels = np.count_nonzero(mask_uint8)
            area_cm2 = round(area_pixels / (PIXELS_PER_CM**2), 2)
            x, y, w, h = cv2.boundingRect(mask_uint8)
            width_cm = round(w / PIXELS_PER_CM, 2)
        except Exception as e:
            print(f"Segmentation Error: {e}")
    
    # --- D. DEPTH ESTIMATION ---
    depth_score = 0.0
    if midas_model:
        try:
            input_batch = midas_transform(img_np).to(device)
            with torch.no_grad():
                prediction = midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            depth_score = np.mean(depth_norm)
            
            depth_colored = cv2.applyColorMap(np.uint8(255 * depth_norm), cv2.COLORMAP_INFERNO)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            result['depth_data'] = array_to_base64(depth_colored)
        except Exception as e:
            print(f"Depth Estimation Error: {e}")
    
    # --- E. LIME ---
    if lime_explainer:
        try:
            def predict_lime_fn(images):
                batch = efficientnet.preprocess_input(np.array(images))
                return tf_model.predict(batch, verbose=0)
            
            lime_exp = lime_explainer.explain_instance(
                np.array(img_tf), 
                predict_lime_fn,
                top_labels=1, 
                hide_color=0, 
                num_samples=100 
            )
            temp_lime, mask_lime = lime_exp.get_image_and_mask(
                lime_exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False
            )
            lime_boundary = mark_boundaries(temp_lime/255.0, mask_lime, color=(1, 1, 0))
            lime_uint8 = (lime_boundary * 255).astype(np.uint8)
            result['lime_data'] = array_to_base64(lime_uint8)
        except Exception as e:
            print(f"LIME Error: {e}")
            result['lime_data'] = None
    
    # --- F. LLM REPORT ---
    ai_results = {
        "Classification": pred_label,
        "Confidence": f"{confidence}%",
        "Wound Area": f"{area_cm2} cm2",
        "Max Width": f"{width_cm} cm",
        "Relative Depth Index": depth_score
    }
    
    result['metrics'] = ai_results
    result['llm_report'] = generate_llm_report(symptoms, ai_results)
    
    return result
