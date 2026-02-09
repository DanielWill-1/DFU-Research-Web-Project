"""
DFU Classifier Web App - Flask Backend
Uses threading for background processing (no RQ/Redis/Celery)
"""
import os
import threading
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from job_manager import job_manager
from ml_pipeline import (
    initialize_models,
    load_and_preprocess_image,
    classify_image,
    generate_gradcam,
    generate_lime,
    segment_wound,
    estimate_depth,
    generate_llm_report,
    image_to_base64,
    cfg
)

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================
# INITIALIZE MODELS AT STARTUP
# ==========================================
initialize_models()

# ==========================================
# CACHE FOR COMPLETED RESULTS (in-memory)
# ==========================================
results_cache = {}  # {job_id: results}

# ==========================================
# BACKGROUND PROCESSING FUNCTION
# ==========================================
def process_image_task(job_id, file_path, symptoms_dict):
    """
    Complete ML pipeline in background thread.
    Stores results in cache for display after reload.
    """
    try:
        job_manager.mark_running(job_id)
        
        # ===== STAGE 1: LOAD & PREPROCESS =====
        job_manager.update_progress(job_id, 10, "Loading and preprocessing image...")
        img_array, img_np, img_pil = load_and_preprocess_image(file_path)
        img_base64 = image_to_base64(img_pil)
        
        # ===== STAGE 2: CLASSIFICATION =====
        job_manager.update_progress(job_id, 25, "Running classification...")
        prediction, confidence, prob_dict = classify_image(img_array)
        pred_idx = cfg.CLASS_NAMES.index(prediction)
        
        # ===== STAGE 3: GRADCAM =====
        job_manager.update_progress(job_id, 40, "Generating GradCAM heatmap...")
        gradcam_data = generate_gradcam(img_array, img_np, pred_idx)
        
        # ===== STAGE 4: LIME =====
        job_manager.update_progress(job_id, 50, "Generating LIME explanation...")
        lime_data = generate_lime(img_pil, pred_idx)
        
        # ===== STAGE 5: SEGMENTATION =====
        job_manager.update_progress(job_id, 70, "Segmenting wound region...")
        seg_data, metrics = segment_wound(img_np)
        
        # ===== STAGE 6: DEPTH ESTIMATION =====
        job_manager.update_progress(job_id, 85, "Computing depth map...")
        depth_data, depth_score = estimate_depth(img_np)
        if depth_score > 0:
            metrics['Relative Depth Index'] = f"{depth_score}"
        
        # ===== STAGE 7: LLM REPORT =====
        job_manager.update_progress(job_id, 95, "Generating medical report...")
        ai_results = {
            "Classification": prediction,
            "Confidence": f"{confidence}%",
            "Wound Area": metrics.get('Wound Area', 'N/A'),
            "Max Width": metrics.get('Max Width', 'N/A'),
            "Relative Depth Index": metrics.get('Relative Depth Index', 'N/A')
        }
        llm_report = generate_llm_report(symptoms_dict, ai_results)
        
        # ===== COLLECT ALL RESULTS =====
        results = {
            'prediction': prediction,
            'confidence': f"{confidence}",
            'metrics': prob_dict,
            'gradcam_data': gradcam_data,
            'lime_data': lime_data,
            'seg_data': seg_data,
            'depth_data': depth_data,
            'llm_report': llm_report,
            'img_data': img_base64,
            'report_date': datetime.now().strftime("%d %B, %Y at %H:%M"),
            'ai_results': ai_results
        }
        
        # Store results in cache for retrieval after reload
        results_cache[job_id] = results
        
        job_manager.mark_completed(job_id, results)
        print(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        job_manager.mark_failed(job_id, str(e))
        print(f"❌ Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main upload endpoint & results display.
    - POST: Creates job and launches background thread
    - GET: Displays form or retrieves cached results
    """
    context = {}
    
    # Check if we're loading results from a completed job
    job_id = request.args.get('job_id')
    if job_id and job_id in results_cache:
        # Display cached results
        context['results'] = results_cache[job_id]
        print(f"📊 Displaying results for job {job_id}")
        # Optional: Delete from cache after displaying (uncomment if you want)
        # del results_cache[job_id]
    
    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file or file.filename == '':
            context['error'] = "Please upload an image."
            return render_template('index.html', **context), 400
        
        try:
            # Collect symptoms
            symptoms = {
                'redness': 'redness' in request.form,
                'swelling': 'swelling' in request.form,
                'odor': 'odor' in request.form,
                'pain': 'pain' in request.form,
                'discharge': 'discharge' in request.form,
                'fever': 'fever' in request.form
            }
            
            # Save uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Create job
            job_id = job_manager.create_job()
            
            # Launch background thread
            thread = threading.Thread(
                target=process_image_task,
                args=(job_id, file_path, symptoms),
                daemon=True
            )
            thread.start()
            
            context['task_id'] = job_id
            print(f"✅ Job created: {job_id}")
            
        except Exception as e:
            context['error'] = f"Error: {str(e)}"
            print(f"❌ Error: {e}")
    
    return render_template('index.html', **context)

@app.route('/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Poll endpoint for job status.
    Returns JSON with state, progress, and completion redirect URL.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    response = {
        'state': job['state'],
        'progress': job['progress'],
        'status': job['status_message'],
        'current': job['progress']
    }
    
    # If completed, include redirect URL
    if job['state'] == 'completed':
        response['redirect_url'] = f'/?job_id={job_id}'
    
    if job['state'] == 'failed':
        response['error'] = job['error']
    
    return jsonify(response)

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 DFU Classifier Web App Starting")
    print("=" * 60)
    print("✅ Threading-based processing (no RQ/Redis/Celery)")
    print("📁 Upload folder:", os.path.abspath(app.config['UPLOAD_FOLDER']))
    print("🌐 Server: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=False, threaded=True, port=5000)