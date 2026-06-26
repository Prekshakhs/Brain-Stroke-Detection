# File: streamlit_app/app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys
import json
from datetime import datetime
import tempfile
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
sys.path.append('..')
sys.path.append('../src')

try:
    from model_def import ResNetClassifier, load_trained_model
except ImportError:
    st.error("Could not import ResNetClassifier. Make sure you're running from the project root directory.")

# Page configuration
st.set_page_config(
    page_title="Brain Stroke Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    .high-confidence {
        border-color: #28a745;
        background-color: #22853a;
    }
    .medium-confidence {
        border-color: #ffc107;
        background-color: #c2a23e;
    }
    .low-confidence {
        border-color: #dc3545;
        background-color: #73121b;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load the trained stroke detection model (PyTorch)"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Loading model on device: {device}")
        print(f"📂 Model path: {model_path}")
        
        # Load using the robust method from model_def.py
        model = load_trained_model(model_path, num_classes=3, device=device)
        if model:
            model.eval()
            model.to(device)
            return model
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Error loading model: {str(e)}")
        return None

def is_brain_scan(image_array):
    """
    Basic image validation - accepts all image types.
    Only checks for minimum/maximum size.
    """
    try:
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        height, width = gray.shape
        
        # Basic dimension checks only
        if width < 80 or height < 80:
            return False, "Image is too small"
        
        if width > 4096 or height > 4096:
            return False, "Image is too large"
        
        # Accept all other images
        return True, "Image accepted"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def preprocess_image(image_path_or_pil):
    """Preprocess image for PyTorch model (grayscale)"""
    # Convert PIL image to numpy and then to grayscale
    if isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil
    else:
        img = Image.open(image_path_or_pil)
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Resize to 224x224
    img_resized = img_gray.resize((224, 224))
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single channel
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict(model, image_tensor, device="cpu"):
    """Make prediction using PyTorch model"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

def create_confidence_chart(probabilities):
    """Create confidence visualization chart"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create DataFrame for plotly
    df = pd.DataFrame({
        'Class': classes,
        'Probability': probs
    })
    
    # Create color mapping
    colors = ['#28a745' if p == max(probs) else '#6c757d' for p in probs]
    
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        title="Prediction Confidence by Class",
        color='Class',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    # Add percentage labels on bars
    for i, prob in enumerate(probs):
        fig.add_annotation(
            x=classes[i],
            y=prob + 0.02,
            text=f'{prob:.1%}',
            showarrow=False,
            font=dict(size=12, color='black')
        )
    
    return fig

def create_medical_report(prediction_result, image_info):
    """Create a formatted medical report"""
    confidence_level = "High" if prediction_result['confidence'] > 0.8 else "Medium" if prediction_result['confidence'] > 0.6 else "Low"
    
    report = f"""
# 🏥 Brain Stroke Detection Report

**Date & Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Image Information:** {image_info}

## 🎯 Analysis Results

**Primary Diagnosis:** {prediction_result['predicted_class'].replace('_', ' ')}  
**Confidence Level:** {prediction_result['confidence']:.1%} ({confidence_level})

## 📊 Detailed Probabilities
"""
    
    for class_name, prob in prediction_result['probabilities'].items():
        report += f"- **{class_name.replace('_', ' ')}:** {prob:.1%}\n"
    
    report += f"""
## 💡 Recommendation
"""
    
    if prediction_result['confidence'] > 0.8:
        report += "High confidence prediction - recommend clinical correlation and expert review."
    elif prediction_result['confidence'] > 0.6:
        report += "Moderate confidence - additional imaging and clinical evaluation recommended."
    else:
        report += "Low confidence prediction - clinical evaluation and expert radiologist review strongly recommended."
    
    report += """

## ⚠️ Important Notes
- This AI analysis is for diagnostic assistance only
- Clinical correlation and expert radiologist review are essential
- Not a substitute for professional medical diagnosis
- Always consult with qualified medical professionals
- Results should be interpreted by licensed healthcare providers
    """
    return report

def generate_pdf_report(prediction_result, image_info, image_bytes=None):
    """Generate a PDF report with prediction results"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Header style
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    # Normal style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    # Add title
    elements.append(Paragraph("BRAIN STROKE DETECTION SYSTEM", title_style))
    elements.append(Paragraph("AI-Assisted Diagnostic Report", styles['Italic']))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Add timestamp and info
    elements.append(Paragraph("<b>Report Generated:</b> " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), normal_style))
    elements.append(Paragraph(f"<b>Image File:</b> {image_info}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Prediction section
    elements.append(Paragraph("PREDICTION RESULTS", header_style))
    confidence_level = "High" if prediction_result['confidence'] > 0.8 else "Medium" if prediction_result['confidence'] > 0.6 else "Low"
    
    table_data = [
        ["Primary Diagnosis", prediction_result['predicted_class'].replace('_', ' ')],
        ["Confidence Level", f"{prediction_result['confidence']:.1%} ({confidence_level})"],
        ["Model Used", prediction_result.get('model', 'Unknown')]
    ]
    
    table = Table(table_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Probabilities section
    elements.append(Paragraph("DETAILED PROBABILITIES", header_style))
    prob_data = [["Class", "Probability"]]
    
    for class_name, prob in sorted(prediction_result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        prob_data.append([class_name.replace('_', ' '), f"{prob:.2%}"])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Recommendation section
    elements.append(Paragraph("CLINICAL RECOMMENDATION", header_style))
    if prediction_result['confidence'] > 0.8:
        recommendation = "High confidence prediction - recommend clinical correlation and expert review."
    elif prediction_result['confidence'] > 0.6:
        recommendation = "Moderate confidence - additional imaging and clinical evaluation recommended."
    else:
        recommendation = "Low confidence prediction - clinical evaluation and expert radiologist review strongly recommended."
    
    elements.append(Paragraph(recommendation, normal_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Important notes
    elements.append(Paragraph("IMPORTANT NOTES", header_style))
    notes = """
    <br/>&bull; This AI analysis is for diagnostic assistance only<br/>
    &bull; Clinical correlation and expert radiologist review are essential<br/>
    &bull; Not a substitute for professional medical diagnosis<br/>
    &bull; Always consult with qualified medical professionals<br/>
    &bull; Results should be interpreted by licensed healthcare providers<br/>
    &bull; <b>BY<br/>Dr. [Vignesh Prabhu], MD, Radiology Specialist</b>
    """
    elements.append(Paragraph(notes, normal_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Footer
    elements.append(Paragraph("=" * 60, styles['Normal']))
    elements.append(Paragraph(
        "This report was automatically generated by the Brain Stroke Detection System. "
        "For official diagnosis and treatment, please consult qualified medical professionals.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save activation during forward pass"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradient during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        """Generate Class Activation Map"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Calculate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Weight activations
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros_like(activations[0])
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = torch.nn.functional.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()

def apply_gradcam(model, image_tensor, class_idx, device="cpu"):
    """Apply Grad-CAM and get heatmap"""
    try:
        image_tensor = image_tensor.to(device)
        
        # Get the last convolutional layer
        target_layer = model.model.layer4[1].conv2 if hasattr(model.model, 'layer4') else model.model.fc
        
        # Create GradCAM object
        gradcam = GradCAM(model.model, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor, class_idx)
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        return cam_resized
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {str(e)}")
        return None

def create_gradcam_visualization(original_image, cam, class_name):
    """Create visualization with Grad-CAM overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(original_image, cmap='gray', alpha=0.5)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay - {class_name}')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================================

def extract_patient_id(filename):
    """Extract patient ID from filename"""
    # Assumes format: PATIENT_ID_SLICE.jpg
    try:
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            return parts[0]
        return filename
    except:
        return filename

def aggregate_patient_predictions(predictions_list):
    """Aggregate predictions by patient using majority voting"""
    if not predictions_list:
        return {}
    
    # Group by patient
    patient_groups = {}
    for pred in predictions_list:
        patient_id = extract_patient_id(pred['image_name'])
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(pred)
    
    # Majority voting for each patient
    aggregated = {}
    for patient_id, group in patient_groups.items():
        predictions = [p['predicted_class'] for p in group]
        
        # Count votes
        from collections import Counter
        votes = Counter(predictions)
        majority_class = votes.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = np.mean([p['confidence'] for p in group])
        
        aggregated[patient_id] = {
            'predicted_class': majority_class,
            'confidence': avg_confidence,
            'num_slices': len(group),
            'class_distribution': dict(votes)
        }
    
    return aggregated

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_image_metrics(predictions, ground_truth):
    """Compute metrics for image-level predictions"""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        pred_classes = [p['predicted_class'] for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, pred_classes),
            'precision': precision_score(ground_truth, pred_classes, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, pred_classes, average='weighted', zero_division=0),
            'f1': f1_score(ground_truth, pred_classes, average='weighted', zero_division=0)
        }
        
        return metrics
    except:
        return None

def compute_patient_metrics(patient_preds, ground_truth):
    """Compute metrics for patient-level predictions"""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        pred_classes = [p['predicted_class'] for p in patient_preds.values()]
        
        metrics = {
            'accuracy': accuracy_score(ground_truth, pred_classes),
            'precision': precision_score(ground_truth, pred_classes, average='weighted', zero_division=0),
            'recall': recall_score(ground_truth, pred_classes, average='weighted', zero_division=0),
            'f1': f1_score(ground_truth, pred_classes, average='weighted', zero_division=0)
        }
        
        return metrics
    except:
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Stroke Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Stroke Classification from CT/MRI Scans**")
    
    # Sidebar
    st.sidebar.header("🎛️ Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["Single Image Analysis", "Batch Processing", "Model Information", "About"]
    )
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    
    # Look for available models - try multiple paths
    possible_paths = [
        "../models",
        "../../models",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
        os.path.abspath(os.path.join(os.getcwd(), "..", "models"))
    ]
    
    model_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            model_dir = path
            break
    
    available_models = []
    
    if model_dir and os.path.exists(model_dir):
        available_models = [f for f in os.listdir(model_dir) if f.endswith(('.pth', '.ckpt'))]
    else:
        # Debug info
        st.sidebar.warning(f"⚠️ Could not find models directory. Checked paths: {possible_paths}")
    
    if not available_models:
        st.sidebar.error(f"❌ No trained models found!")
        if model_dir:
            st.error(f"❌ No model files (.pth or .ckpt) found in: {model_dir}")
        else:
            st.error("❌ Could not locate models directory. Please make sure you have a 'models/' folder with trained model files.")
        st.info("📌 Please train a model first using: `python train_model.py`")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        help="Choose which trained model to use for predictions"
    )
    
    model_path = os.path.join(model_dir, selected_model)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error(f"❌ Unable to load the selected model: {selected_model}")
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model}")
    
    # Main application modes
    if app_mode == "Single Image Analysis":
        st.header("📤 Upload Brain Scan for Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a brain scan image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload CT or MRI scan images"
        )
        
        if uploaded_file is not None:
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Original Image")
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                # Validate if it's a brain scan - convert to BGR format for OpenCV
                image_array = np.array(image)
                if image.mode == 'RGB':
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                elif image.mode == 'L':
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                is_valid, validation_message = is_brain_scan(image_array)
                
                if not is_valid:
                    st.error(f"❌ Invalid Image: {validation_message}")
                    st.warning("""
                    ⚠️ **Only CT or MRI brain scan images are accepted.**
                    
                    Please upload:
                    - CT (Computed Tomography) brain scans
                    - MRI (Magnetic Resonance Imaging) brain scans
                    - Can be grayscale or colorized
                    - Standard medical imaging formats (PNG, JPG, JPEG, BMP, TIFF)
                    
                    Do NOT upload:
                    - Natural images (cars, animals, landscapes, etc.)
                    - Other medical images (X-rays of other body parts, ultrasounds, etc.)
                    - Screenshots or diagrams
                    - Low-quality or corrupted images
                    """)
                else:
                    st.success("✅ Valid brain scan image detected!")
                    st.image(image, caption="Uploaded Brain Scan", use_column_width=True)
                    
                    # Image information
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {image.size}")
                    st.write(f"**Format:** {image.format}")
                    
                    # Analysis button
                    analyze_button = st.button("🔍 Analyze Image", type="primary")
                
            with col2:
                st.subheader("🎯 Analysis Results")
                
                if is_valid and analyze_button:
                    with st.spinner("Analyzing image..."):
                        try:
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            
                            # Preprocess image
                            img_tensor = preprocess_image(uploaded_file)
                            
                            # Make prediction
                            class_idx, confidence, probs = predict(model, img_tensor, device)
                            
                            # Class names
                            class_names = ['Hemorrhagic', 'Ischemic', 'NoStroke']
                            predicted_class = class_names[class_idx]
                            
                            # Create result dictionary
                            result = {
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'class_index': class_idx,
                                'probabilities': {
                                    class_names[i]: float(probs[i]) for i in range(len(class_names))
                                }
                            }
                            
                            # Determine confidence level and styling
                            if confidence > 0.8:
                                confidence_class = "high-confidence"
                                confidence_emoji = "🟢"
                            elif confidence > 0.6:
                                confidence_class = "medium-confidence"
                                confidence_emoji = "🟡"
                            else:
                                confidence_class = "low-confidence"
                                confidence_emoji = "🔴"
                            
                            # Display prediction box
                            st.markdown(f"""
                            <div class="prediction-box {confidence_class}">
                                <h3>{confidence_emoji} Prediction Results</h3>
                                <p><strong>Predicted Class:</strong> {result['predicted_class']}</p>
                                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence chart
                            st.subheader("📊 Confidence Distribution")
                            chart = create_confidence_chart(result['probabilities'])
                            st.plotly_chart(chart, use_container_width=True)
                            
                            # Detailed probabilities
                            st.subheader("📋 Detailed Probabilities")
                            prob_df = pd.DataFrame([
                                {"Class": k, "Probability": f"{v:.2%}"}
                                for k, v in result['probabilities'].items()
                            ])
                            st.dataframe(prob_df, use_container_width=True)
                            
                            # Grad-CAM Visualization
                            st.subheader("🔍 Model Interpretability - Grad-CAM Heatmap")
                            st.info("💡 Red areas indicate regions the model focused on for this prediction")
                            
                            with st.spinner("Generating Grad-CAM visualization..."):
                                try:
                                    # Convert PIL image to numpy for display
                                    img_np = np.array(image.convert('L'))
                                    
                                    # Generate Grad-CAM
                                    cam = apply_gradcam(model, img_tensor, class_idx, device)
                                    
                                    if cam is not None:
                                        # Create visualization
                                        fig = create_gradcam_visualization(img_np, cam, predicted_class)
                                        st.pyplot(fig)
                                        
                                        st.success("✅ Grad-CAM visualization generated successfully!")
                                    else:
                                        st.warning("⚠️ Could not generate Grad-CAM visualization")
                                
                                except Exception as e:
                                    st.warning(f"⚠️ Grad-CAM generation failed: {str(e)}")
                            
                            # Medical report
                            st.subheader("🏥 Medical Report")
                            report = create_medical_report(result, f"{uploaded_file.name} ({image.size})")
                            st.markdown(report)
                            
                            # Download report
                            report_data = {
                                'timestamp': datetime.now().isoformat(),
                                'image_name': uploaded_file.name,
                                'prediction': result,
                                'model_used': selected_model
                            }
                            
                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📄 Download Report (JSON)",
                                    data=json.dumps(report_data, indent=2),
                                    file_name=f"stroke_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            
                            with col2:
                                try:
                                    pdf_buffer = generate_pdf_report(result, uploaded_file.name)
                                    st.download_button(
                                        label="📕 Download Report (PDF)",
                                        data=pdf_buffer,
                                        file_name=f"stroke_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.warning(f"⚠️ PDF generation not available: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"❌ Failed to analyze the image: {str(e)}")
    
    elif app_mode == "Batch Processing":
        st.header("📚 Batch Image Processing")
        
        st.info("💡 This feature allows you to analyze multiple images at once with patient-level aggregation.")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose multiple brain scan images...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple CT or MRI scan images for batch analysis"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Analyze All Images", type="primary"):
                progress_bar = st.progress(0)
                results = []
                invalid_images = []
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Validate image first - convert to BGR format for OpenCV
                        temp_image = Image.open(uploaded_file)
                        image_array = np.array(temp_image)
                        if temp_image.mode == 'RGB':
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        elif temp_image.mode == 'L':
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                        
                        is_valid, validation_message = is_brain_scan(image_array)
                        
                        if not is_valid:
                            invalid_images.append((uploaded_file.name, validation_message))
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            continue
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Preprocess and predict
                        img_tensor = preprocess_image(uploaded_file)
                        class_idx, confidence, probs = predict(model, img_tensor, device)
                        
                        class_names = ['Hemorrhagic', 'Ischemic', 'NoStroke']
                        result = {
                            'image_name': uploaded_file.name,
                            'predicted_class': class_names[class_idx],
                            'confidence': confidence,
                            'probabilities': {class_names[j]: float(probs[j]) for j in range(len(class_names))}
                        }
                        results.append(result)
                    except Exception as e:
                        st.warning(f"Failed to process {uploaded_file.name}: {str(e)}")
                
                # Show validation errors if any
                if invalid_images:
                    st.warning(f"⚠️ {len(invalid_images)} invalid image(s) skipped:")
                    for filename, reason in invalid_images:
                        st.write(f"  • {filename}: {reason}")
                
                if results:
                    st.success(f"✅ Processed {len(results)} images successfully!")
                    
                    # ========== IMAGE-LEVEL RESULTS ==========
                    st.subheader("📊 Image-Level Analysis")
                    
                    predictions_count = {}
                    for result in results:
                        pred_class = result['predicted_class']
                        predictions_count[pred_class] = predictions_count.get(pred_class, 0) + 1
                    
                    summary_df = pd.DataFrame([
                        {"Diagnosis": k.replace('_', ' '), "Count": v, "Percentage": f"{(v/len(results)*100):.1f}%"}
                        for k, v in predictions_count.items()
                    ])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Pie chart
                        fig_pie = px.pie(
                            summary_df,
                            names='Diagnosis',
                            values='Count',
                            title='Distribution of Image-Level Predictions',
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Detailed image results
                    st.subheader("📋 Detailed Image Results")
                    
                    detailed_df = pd.DataFrame([
                        {
                            "Image": result['image_name'],
                            "Prediction": result['predicted_class'].replace('_', ' '),
                            "Confidence": f"{result['confidence']:.2%}"
                        }
                        for result in results
                    ])
                    
                    st.dataframe(detailed_df, use_container_width=True)
                    
                    # ========== PATIENT-LEVEL RESULTS ==========
                    st.subheader("👥 Patient-Level Aggregation (Majority Voting)")
                    
                    with st.spinner("Aggregating predictions by patient..."):
                        patient_preds = aggregate_patient_predictions(results)
                        
                        if patient_preds:
                            st.success(f"✅ Aggregated {len(results)} images into {len(patient_preds)} patient(s)")
                            
                            # Patient-level statistics
                            patient_pred_count = {}
                            for patient_data in patient_preds.values():
                                pred_class = patient_data['predicted_class']
                                patient_pred_count[pred_class] = patient_pred_count.get(pred_class, 0) + 1
                            
                            patient_summary_df = pd.DataFrame([
                                {"Diagnosis": k.replace('_', ' '), "Patient Count": v, "Percentage": f"{(v/len(patient_preds)*100):.1f}%"}
                                for k, v in patient_pred_count.items()
                            ])
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig_pie_patient = px.pie(
                                    patient_summary_df,
                                    names='Diagnosis',
                                    values='Patient Count',
                                    title='Distribution of Patient-Level Predictions',
                                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                                )
                                st.plotly_chart(fig_pie_patient, use_container_width=True)
                            
                            with col2:
                                st.dataframe(patient_summary_df, use_container_width=True)
                            
                            # Detailed patient results
                            st.subheader("📋 Detailed Patient Results")
                            
                            patient_details = []
                            for patient_id, data in patient_preds.items():
                                patient_details.append({
                                    "Patient ID": patient_id,
                                    "Diagnosis": data['predicted_class'].replace('_', ' '),
                                    "Confidence": f"{data['confidence']:.2%}",
                                    "Number of Slices": data['num_slices'],
                                    "Prediction Distribution": str(data['class_distribution'])
                                })
                            
                            patient_details_df = pd.DataFrame(patient_details)
                            st.dataframe(patient_details_df, use_container_width=True)
                            
                            # Comparison: Image-level vs Patient-level
                            st.subheader("📊 Comparison: Image-Level vs Patient-Level")
                            
                            comparison_data = {
                                'Metric': ['Total Count', 'Hemorrhagic', 'Ischemic', 'NoStroke'],
                                'Image-Level': [
                                    len(results),
                                    predictions_count.get('Hemorrhagic', 0),
                                    predictions_count.get('Ischemic', 0),
                                    predictions_count.get('NoStroke', 0)
                                ],
                                'Patient-Level': [
                                    len(patient_preds),
                                    patient_pred_count.get('Hemorrhagic', 0),
                                    patient_pred_count.get('Ischemic', 0),
                                    patient_pred_count.get('NoStroke', 0)
                                ]
                            }
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Bar chart comparison
                            fig_comparison = px.bar(
                                comparison_df[comparison_df['Metric'] != 'Total Count'],
                                x='Metric',
                                y=['Image-Level', 'Patient-Level'],
                                barmode='group',
                                title='Prediction Distribution Comparison',
                                labels={'value': 'Count', 'variable': 'Analysis Level'}
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # ========== EXPORT ==========
                    st.subheader("💾 Export Results")
                    
                    # Create comprehensive report
                    batch_report = {
                        'timestamp': datetime.now().isoformat(),
                        'analysis_type': 'batch_with_patient_aggregation',
                        'image_level': {
                            'total_images': len(results),
                            'predictions': results,
                            'summary': predictions_count
                        },
                        'patient_level': {
                            'total_patients': len(patient_preds),
                            'predictions': {k: {
                                'predicted_class': v['predicted_class'],
                                'confidence': float(v['confidence']),
                                'num_slices': v['num_slices'],
                                'class_distribution': v['class_distribution']
                            } for k, v in patient_preds.items()},
                            'summary': patient_pred_count
                        },
                        'model_used': selected_model
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download Full Report (JSON)",
                            data=json.dumps(batch_report, indent=2),
                            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # CSV export - Image level
                        csv_image_level = pd.DataFrame(results).to_csv(index=False)
                        st.download_button(
                            label="📊 Download Image-Level CSV",
                            data=csv_image_level,
                            file_name=f"image_level_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # CSV export - Patient level
                        csv_patient_level = patient_details_df.to_csv(index=False)
                        st.download_button(
                            label="👥 Download Patient-Level CSV",
                            data=csv_patient_level,
                            file_name=f"patient_level_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ No images could be processed successfully.")
    
    elif app_mode == "Model Information":
        st.header("🤖 Model Information")
        
        if model and model.model:
            # Model architecture
            st.subheader("🏗️ Model Architecture")
            
            try:
                # Get model summary as string
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = StringIO()
                model.model.summary()
                sys.stdout = old_stdout
                summary_string = buffer.getvalue()
                
                st.code(summary_string, language='text')
                
                # Model statistics
                st.subheader("📊 Model Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                # Count parameters for PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                
                with col1:
                    st.metric("Total Parameters", f"{total_params:,}")
                
                with col2:
                    st.metric("Trainable Parameters", f"{trainable_params:,}")
                
                with col3:
                    st.metric("Non-trainable Parameters", f"{non_trainable_params:,}")
                
                # Model configuration
                st.subheader("⚙️ Model Configuration")
                
                config_data = {
                    "Input Shape": str(model.img_size),
                    "Number of Classes": model.num_classes,
                    "Class Names": ", ".join(model.class_names),
                    "Model File": selected_model
                }
                
                config_df = pd.DataFrame([
                    {"Parameter": k, "Value": v} for k, v in config_data.items()
                ])
                
                st.dataframe(config_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying model information: {e}")
    
    elif app_mode == "About":
        st.header("ℹ️ About Brain Stroke Detection System")
        
        st.markdown("""
        ## 🧠 Overview
        
        This Brain Stroke Detection System uses advanced Convolutional Neural Networks (CNN) to automatically 
        detect and classify brain strokes from CT/MRI medical imaging data into three categories:
        
        - **No Stroke**: Normal brain scans without stroke indicators
        - **Hemorrhagic Stroke**: Bleeding in the brain
        - **Ischemic Stroke**: Blood clot blocking brain blood supply
        
        ## 🎯 Key Features
        
        - **AI-Powered Analysis**: Uses state-of-the-art deep learning models
        - **Real-time Predictions**: Fast analysis suitable for clinical use
        - **High Accuracy**: Trained on medical imaging datasets
        - **User-friendly Interface**: Easy-to-use web interface
        - **Detailed Reports**: Comprehensive analysis reports
        - **Batch Processing**: Analyze multiple images simultaneously
        
        ## 🔬 Technology Stack
        
        - **Deep Learning**: TensorFlow/Keras
        - **Computer Vision**: OpenCV, PIL
        - **Web Interface**: Streamlit
        - **Visualization**: Plotly, Matplotlib
        - **Data Processing**: NumPy, Pandas
        
        ## ⚠️ Important Disclaimers
        
        - This system is designed for **research and educational purposes**
        - **NOT a substitute** for professional medical diagnosis
        - Always consult qualified medical professionals
        - Clinical validation required for medical use
        - Ensure HIPAA compliance when handling patient data
        
        ## 📚 Usage Instructions
        
        1. **Single Image Analysis**: Upload one brain scan for immediate analysis
        2. **Batch Processing**: Upload multiple images for batch analysis
        3. **View Results**: Review predictions, confidence scores, and detailed reports
        4. **Download Reports**: Save analysis results for future reference
        
        ## 🤝 Support
        
        For technical support, feature requests, or bug reports, please contact the development team.
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
        """)

if __name__ == "__main__":
    main()