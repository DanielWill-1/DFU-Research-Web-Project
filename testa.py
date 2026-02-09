"""
DFU Classifier Web App - Flask Backend
Uses threading for background processing (no RQ/Redis/Celery)
"""
import os
import threading
import base64
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file, abort
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
        # Store numeric depth score and also a human readable Depth Index (percentage)
        results_depth_score = float(depth_score) if depth_score is not None else 0.0
        depth_display = f"{results_depth_score*100:.1f}%" if results_depth_score > 0 else "N/A"
        if results_depth_score > 0:
            metrics['Relative Depth Index'] = depth_display

        # ===== STAGE 7: LLM REPORT =====
        job_manager.update_progress(job_id, 95, "Generating medical report...")
        ai_results = {
            "Classification": prediction,
            "Confidence": f"{confidence}%",
            "Wound Area": metrics.get('Wound Area', 'N/A'),
            "Max Width": metrics.get('Max Width', 'N/A'),
            "Relative Depth Index": metrics.get('Relative Depth Index', depth_display)
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
            'depth_score': results_depth_score,        # numeric depth score 0..1
            'llm_report': llm_report,
            'img_data': img_base64,
            'report_date': datetime.now().strftime("%d %B, %Y at %H:%M"),
            'ai_results': ai_results,
            'symptoms': symptoms_dict        # store for PDF/report
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
        context['job_id'] = job_id       # <-- added so template can link to /download-report/<job_id>
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

def generate_pdf_report(job_id, results):
	"""
	Generate PDF (BytesIO) from results. Local ReportLab imports to keep runtime safe.
	Returns BytesIO or None on failure.
	"""
	try:
		from reportlab.lib.pagesizes import letter
		from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
		from reportlab.lib.units import inch
		from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
		from reportlab.lib import colors

		buf = BytesIO()
		doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
		styles = getSampleStyleSheet()
		elements = []

		# Title
		title_style = ParagraphStyle('title', parent=styles['Heading1'], alignment=1)
		elements.append(Paragraph("🩺 DFU Classifier Analysis Report", title_style))
		elements.append(Spacer(1, 0.2*inch))

		# Metadata
		elements.append(Paragraph(f"<b>Job ID:</b> {job_id}", styles['Normal']))
		elements.append(Paragraph(f"<b>Report Date:</b> {results.get('report_date', 'N/A')}", styles['Normal']))
		elements.append(Spacer(1, 0.2*inch))

		# Classification
		elements.append(Paragraph("<b>Classification</b>", styles['Heading2']))
		elements.append(Paragraph(f"Prediction: {results.get('prediction', 'N/A')}", styles['Normal']))
		elements.append(Paragraph(f"Confidence: {results.get('confidence', 'N/A')}%", styles['Normal']))
		elements.append(Spacer(1, 0.1*inch))

		# AI metrics table
		ai = results.get('ai_results', {})
		table_data = [["Metric", "Value"]]
		for k, v in ai.items():
			table_data.append([k, str(v)])
		table = Table(table_data, colWidths=[3.0*inch, 3.0*inch])
		table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey),
								   ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)]))
		elements.append(table)
		elements.append(Spacer(1, 0.2*inch))

		# Symptoms
		elements.append(Paragraph("<b>Reported Symptoms</b>", styles['Heading2']))
		symptoms = results.get('symptoms', {})
		if symptoms:
			for k, v in symptoms.items():
				elements.append(Paragraph(f"{k.capitalize()}: {'Yes' if v else 'No'}", styles['Normal']))
		else:
			elements.append(Paragraph("No symptoms provided.", styles['Normal']))
		elements.append(Spacer(1, 0.2*inch))

		# Helper to add images from base64
		def add_image(b64, caption=None, max_width=4.5*inch):
			if not b64:
				return
			try:
				img_bytes = BytesIO(base64.b64decode(b64))
				img = RLImage(img_bytes)
				# Restrict width
				img.drawWidth = min(img.drawWidth, max_width)
				# Maintain aspect ratio
				img.drawHeight = img.drawHeight * (img.drawWidth / img.drawWidth) if img.drawWidth else img.drawHeight
				if caption:
					elements.append(Paragraph(f"<b>{caption}</b>", styles['Normal']))
				elements.append(img)
				elements.append(Spacer(1, 0.1*inch))
			except Exception as e:
				print(f"Warning: could not add image to PDF: {e}")

		add_image(results.get('img_data'), "Original Image")
		add_image(results.get('seg_data'), "Segmentation Mask")
		add_image(results.get('depth_data'), "Depth Map (MiDaS)")
		add_image(results.get('gradcam_data'), "GradCAM Heatmap")
		add_image(results.get('lime_data'), "LIME Explanation")

		# LLM report
		if results.get('llm_report'):
			elements.append(Paragraph("<b>AI Medical Report</b>", styles['Heading2']))
			elements.append(Paragraph(results.get('llm_report'), styles['Normal']))
			elements.append(Spacer(1, 0.1*inch))

		doc.build(elements)
		buf.seek(0)
		return buf

	except Exception as e:
		print(f"PDF generation failed: {e}")
		return None

@app.route('/download-report/<job_id>', methods=['GET'])
def download_report(job_id):
	"""Return a generated PDF report for job_id, or 404/500 JSON on error."""
	results = results_cache.get(job_id)
	if not results:
		return jsonify({'error': 'Results not found for this job ID'}), 404

	pdf_buf = generate_pdf_report(job_id, results)
	if not pdf_buf:
		return jsonify({'error': 'Failed to generate PDF'}), 500

	return send_file(pdf_buf, mimetype='application/pdf', as_attachment=True,
					 download_name=f"dfu_report_{job_id}.pdf")

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