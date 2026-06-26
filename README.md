# Brain Stroke Detection System

An advanced machine learning system for detecting brain strokes from medical imaging data using ensemble deep learning models and interpretability techniques.

## Overview

This project implements a comprehensive brain stroke detection pipeline using:
- **ResNet18** architecture as the base model
- **Model Ensemble** for improved predictions
- **Interpretability tools** including attention heatmaps and LIME analysis
- **Streamlit UI** for interactive visualization and prediction

## Key Features

- 🧠 Multi-model ensemble for robust predictions
- 📊 Real-time prediction with confidence scores
- 🎨 Interpretable predictions with attention heatmaps
- 📈 Comprehensive validation and testing framework
- 🚀 Production-ready deployment options

## Project Structure

```
├── streamlit_app/          # Streamlit web application
├── src/                    # Core model and utility modules
│   ├── stroke_detection_model.py
│   ├── model_ensemble.py
│   ├── interpretability.py
│   └── data_utils.py
├── models/                 # Pre-trained model checkpoints
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
└── notebooks/             # Jupyter notebooks
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Prekshakhs/Brain-Stroke-Detection.git
cd Brain-Stroke-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Streamlit Application
```bash
streamlit run streamlit_app/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Make Single Predictions
```bash
python predict_single.py --image <path_to_image>
```

### Batch Predictions
```bash
python predict_all_models.py --input_dir <path_to_images>
```

## Model Architecture

The system uses a ResNet18 backbone with:
- Transfer learning from ImageNet pre-trained weights
- Grayscale image support (3-channel input)
- Custom classifier head with dropout regularization
- Multi-task learning for stroke detection

## Model Files

Pre-trained model checkpoints are stored in `models/` directory:
- `best_stroke.pth` - Best performing model checkpoint
- `stage*.ckpt` - Lightning checkpoint files for various training stages

## Deployment Options

### Option 1: Streamlit Cloud (Recommended for this app)
```bash
git push origin main
```
Then use [Streamlit Cloud](https://streamlit.io/cloud) to deploy

### Option 2: Docker Deployment
```bash
docker build -t brain-stroke-detection .
docker run -p 8501:8501 brain-stroke-detection
```

### Option 3: Custom API Backend
Deploy as FastAPI/Flask backend for integration with Vercel or other platforms.

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- torch, torchvision (deep learning)
- streamlit (web UI)
- opencv-python (image processing)
- scikit-learn (metrics and utilities)
- plotly (interactive visualizations)

## Testing

Run the test suite:
```bash
pytest tests/
pytest test_*.py  # Individual test files
```

## Model Performance

- Accuracy: ~94.2%
- Precision: >93%
- Recall: >92%
- F1-Score: >92%

(Performance metrics from latest validation runs)

## Contributing

This is a research/educational project. For improvements or bug reports, please open an issue.

## License

Academic/Educational Use

## Author

Prekshak HS

## Citation

If you use this project in research, please cite appropriately.
