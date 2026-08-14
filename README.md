# DFU Website Project

A Flask web app for Diabetic Foot Ulcer (DFU) image analysis. Upload a wound photo, report symptoms, and get an AI analysis: classification, GradCAM and LIME explanations, wound segmentation, depth estimation, and an LLM-generated medical report (PDF download included).

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Routes](#api-routes)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Classification** - EfficientNet model classifies the wound as `Both`, `Infection`, `Ischaemia`, or `None`
- **GradCAM & LIME** - XAI visual explanations of the classification
- **Segmentation** - PyTorch U-Net (EfficientNet-B0 encoder) segments the wound and estimates area/width
- **Depth Estimation** - MiDaS produces a depth map and relative depth index
- **LLM Report** - Groq generates a medical report from symptoms + AI findings
- **PDF Report** - Downloadable PDF summary via ReportLab
- **Background Processing** - Threading-based job manager with progress polling (no Redis/Celery needed)

## Prerequisites

- **Python 3.9 or higher**
- **pip** - Python package manager

Verify your installation:

```bash
python --version
pip --version
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/voiceform/DFU-website-Project.git
   cd DFU-website-Project
   ```

2. **Create and activate a virtual environment** (recommended)

   On Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)

   Create a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Without a key, the app still works but the LLM report section shows "LLM Analysis Unavailable".

## Running the Application

```bash
python main.py
```

The app starts at `http://localhost:5000`. Open that URL in your browser, upload an image, tick any symptoms, and submit. Results appear when the background job completes; use the "Download PDF Report" button to save a copy.

## API Routes

| Method | Route                      | Description                                          |
| ------ | -------------------------- | ---------------------------------------------------- |
| GET/POST | `/`                      | Upload form; POST starts a job, GET with `?job_id=` shows cached results |
| GET    | `/job-status/<job_id>`     | JSON poll endpoint: `state`, `progress`, `status`, and `redirect_url` when done |
| GET    | `/download-report/<job_id>`| Downloads the PDF report for a completed job        |

## Project Structure

```
DFU-website-Project/
├── main.py                        # Flask app entry point (routes + background jobs)
├── ml_pipeline.py                 # ML models and inference pipeline (loaded once at startup)
├── job_manager.py                 # Thread-safe background job manager
├── final_dfu_model_weighted.h5    # TensorFlow classification weights
├── new_unet_seg.pth               # PyTorch segmentation weights
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (GROQ_API_KEY) - not committed
├── templates/
│   └── index.html                 # Main UI template
├── uploads/                       # Runtime uploads (created automatically, gitignored)
└── old/                           # Archived legacy files (old scripts, models, templates, images)
```

The `old/` folder holds superseded versions (old entry scripts, previous model weights, unused templates and sample images). It is not needed to run the app.

## Configuration

Settings live in `ml_pipeline.py` (`Config` class) and `main.py`:

- `SECRET_KEY` - Flask secret (change for production)
- `MAX_CONTENT_LENGTH` - 50 MB upload limit
- `UPLOAD_FOLDER` - where uploads are stored (`uploads/`)
- `TF_MODEL_PATH` / `TORCH_SEG_PATH` - model weight locations
- `PIXELS_PER_CM` - calibration for wound area/width metrics
- `GROQ_API_KEY` - loaded from `.env`
- Port - change `app.run(port=5000)` in `main.py`

## Troubleshooting

### Port Already in Use

Change the port in `main.py`:

```python
app.run(debug=False, threaded=True, port=5001)
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Model Not Found

`main.py` prints which files it looks for at startup. Make sure `final_dfu_model_weighted.h5` and `new_unet_seg.pth` are in the project root.

### MiDaS Download

MiDaS weights are downloaded from `torch.hub` on first run - an internet connection is required the first time.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

[Add your license information here]
