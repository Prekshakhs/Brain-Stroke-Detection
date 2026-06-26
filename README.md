# Brain Stroke Detection and Identification

An advanced machine learning system for detecting and identifying brain strokes from CT and MRI medical imaging data using Convolutional Neural Networks (CNNs) and interpretability techniques.

This project was developed as part of the Bachelor of Engineering in Computer Science and Design at **Canara Engineering College**.

## Authors
- Shivjith Shetty
- Prekshak HS
- Swasthik Gambheer
- Vignesh Prabhu

## Overview

Brain stroke is a critical medical emergency that requires timely and accurate diagnosis to reduce the risk of severe neurological damage and mortality. Traditional diagnostic methods depend on expert interpretation of medical imaging, which can be time-consuming and prone to human error. 

This project proposes a deep learning-based diagnostic model utilizing Convolutional Neural Networks (CNNs) to automatically detect and identify brain strokes from medical imaging data. The model is trained on a labeled dataset containing stroke and non-stroke brain images and is capable of learning complex spatial features for accurate classification. By integrating AI into healthcare, this system serves as an assistive diagnostic tool for medical professionals, enabling faster and more reliable detection of strokes.

## Key Features

- 🧠 **CNN-based Architecture**: Custom deep learning model for robust predictions.
- 📊 **Real-time Prediction**: Fast inference with confidence scores for single and batch image processing.
- 🎨 **Interpretable Predictions**: Grad-CAM attention heatmaps to highlight clinically relevant regions influencing the model's decision.
- 📉 **Comprehensive Preprocessing**: Automatic resizing, normalization, and data augmentation.
- 🚀 **Streamlit Web Interface**: User-friendly GUI for easy interaction by medical professionals.

## Project Structure

```text
├── streamlit_app/          # Streamlit web application
├── src/                    # Core model and utility modules
│   ├── stroke_detection_model.py
│   ├── model_ensemble.py
│   ├── interpretability.py
│   └── data_utils.py
├── models/                 # Pre-trained model checkpoints
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
└── notebooks/              # Jupyter notebooks
```

## Proposed System Workflow

1. **Data Collection & Preprocessing**: Images are resized to 224x224 pixels, normalized, and augmented (rotation, flipping, zooming) to increase dataset diversity.
2. **Model Training**: The CNN model extracts features using convolution and pooling layers, followed by dense layers for classification.
3. **Prediction Engine**: Generates class probabilities (Stroke vs. Non-Stroke) and confidence scores.
4. **Model Interpretability**: Grad-CAM extracts gradients from the final convolution layer to generate heatmaps.
5. **Deployment**: Streamlit interface allows users to upload scans and view the processed image, predicted class, and heatmap visualization.

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
The application will open in your default browser at `http://localhost:8501`.

### Make Single Predictions
```bash
python predict_single.py --image <path_to_image>
```

### Batch Predictions
```bash
python predict_all_models.py --input_dir <path_to_images>
```

## Model Performance

The proposed CNN-based model has undergone extensive unit, integration, and system testing, demonstrating robust performance on both CT and MRI images. 

- **Accuracy**: 93% - 95% (Validation runs reached ~94.2%)
- **Precision**: > 93%
- **Recall**: > 92%
- **F1-Score**: > 92%

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Data Manipulation**: NumPy, Pandas
- **Machine Learning Utilities**: Scikit-learn
- **Frontend**: Streamlit, HTML/CSS
- **Data Visualization**: Matplotlib, Plotly

## Future Scope

- **Integration of Larger Datasets**: Training on multi-hospital datasets to improve generalization.
- **Multi-Class Stroke Classification**: Categorizing strokes into specific subtypes like ischemic, hemorrhagic, and TIA.
- **Incorporation of Clinical Metadata**: Combining patient details (age, symptoms, blood pressure) with imaging data.
- **Edge Deployment**: Real-time processing on edge devices for emergency rooms or ambulances.

## License

Academic/Educational Use
