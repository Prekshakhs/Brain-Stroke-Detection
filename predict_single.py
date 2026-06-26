#!/usr/bin/env python3
"""
Single Image Prediction with Visual Output
Shows image, prediction, and confidence graph
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
from model_def import load_trained_model

CLASS_NAMES = ["Hemorrhagic", "Ischemic", "NoStroke"]
CLASS_COLORS = {
    "Hemorrhagic": "#FF6B6B",
    "Ischemic": "#4ECDC4",
    "NoStroke": "#45B7D1"
}


def predict_and_visualize(image_path, model_path='models/best_stroke.pth', device='cpu'):
    """
    Make prediction and display results with visualization
    """
    
    # Validate inputs
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return None
    
    print("\n" + "="*70)
    print("SINGLE IMAGE PREDICTION WITH VISUALIZATION")
    print("="*70)
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model...")
    try:
        model = load_trained_model(model_path, num_classes=len(CLASS_NAMES), device=device)
        if model is None:
            print("Error: Failed to load model")
            return None
        print("OK: Model loaded")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Load image
    print(f"\nLoading image...")
    try:
        img_pil = Image.open(image_path)
        img_np = np.array(img_pil.convert('L'))  # Convert to grayscale numpy
        print(f"   Size: {img_pil.size}")
        print("OK: Image loaded")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Preprocess for model
    print(f"\nPreprocessing...")
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        print("OK: Preprocessed")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Make prediction
    print(f"\nMaking prediction...")
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        pred_class = CLASS_NAMES[pred_idx.item()]
        conf = confidence.item()
        prob_dict = {CLASS_NAMES[i]: float(probs[0][i].item()) for i in range(len(CLASS_NAMES))}
        
        print("OK: Prediction complete")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Display terminal output
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nPredicted Class: {pred_class}")
    print(f"Confidence: {conf:.2%}")
    
    print(f"\nCLASS PROBABILITIES:")
    print(f"{'Class':<15} {'Probability':<15} {'Confidence':<10}")
    print(f"{'-'*60}")
    for class_name in CLASS_NAMES:
        prob = prob_dict[class_name]
        bar_length = int(prob * 30)
        bar = "#" * bar_length + "-" * (30 - bar_length)
        emoji = "*" if class_name == pred_class else " "
        print(f"{emoji} {class_name:<13} {prob:>12.2%}  {bar}")
    
    print(f"\n{'='*70}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig = plt.figure(figsize=(16, 6))
    
    # 1. Original Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Brain Scan Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Confidence Bar Chart
    ax2 = plt.subplot(1, 3, 2)
    classes = list(prob_dict.keys())
    probs = list(prob_dict.values())
    colors = [CLASS_COLORS[c] for c in classes]
    
    bars = ax2.barh(classes, probs, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold')
    
    # 3. Prediction Summary
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    # Title box
    title_text = f"PREDICTION RESULT\n{'='*30}"
    ax3.text(0.5, 0.95, title_text, transform=ax3.transAxes, 
            fontsize=13, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Main prediction
    pred_text = f"Class: {pred_class}\nConfidence: {conf:.2%}"
    pred_color = CLASS_COLORS[pred_class]
    ax3.text(0.5, 0.75, pred_text, transform=ax3.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=pred_color, alpha=0.7, edgecolor='black', linewidth=2))
    
    # Details
    details_text = f"Image: {os.path.basename(image_path)}\nModel: {os.path.basename(model_path)}"
    ax3.text(0.5, 0.45, details_text, transform=ax3.transAxes,
            fontsize=10, ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Probability details
    prob_details = "\n".join([f"{c}: {prob:.2%}" for c, prob in prob_dict.items()])
    ax3.text(0.5, 0.15, prob_details, transform=ax3.transAxes,
            fontsize=10, ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle(f'Brain Stroke Detection - {pred_class}', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    print("OK: Visualization created")
    print("\nDisplaying image and prediction graph...")
    print("(Close the window to exit)")
    
    plt.show()
    
    return {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'predicted_class': pred_class,
        'confidence': conf,
        'probabilities': prob_dict,
        'model': model_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stroke type with visual output")
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='models/best_stroke.pth', help='Path to model')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    result = predict_and_visualize(args.image, args.model, args.device)
    
    if result:
        print("\nOK: PREDICTION COMPLETE\n")
    else:
        print("\nError: PREDICTION FAILED\n")
        sys.exit(1)
