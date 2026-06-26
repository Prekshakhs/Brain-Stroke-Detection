#!/usr/bin/env python3
"""
Real-Time Prediction using Stage2 Model
stage2-best-epoch=19-val_acc=0.932.ckpt
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from model_def import load_trained_model
import time

CLASS_NAMES = ["Hemorrhagic", "Ischemic", "NoStroke"]
CLASS_COLORS = {
    "Hemorrhagic": "RED",
    "Ischemic": "CYAN",
    "NoStroke": "BLUE"
}


def real_time_predict(image_path, model_path='models/stage2-best-epoch=19-val_acc=0.932.ckpt', device='cpu'):
    """
    Real-time prediction with stage2 model
    """
    
    # Validate inputs
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("\nAvailable models in models/ folder:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith(('.pth', '.ckpt')):
                    size = os.path.getsize(os.path.join('models', f)) / (1024*1024)
                    print(f"   - {f} ({size:.1f} MB)")
        return None
    
    print("\n" + "="*70)
    print("REAL-TIME PREDICTION - STAGE2 MODEL")
    print("="*70)
    
    start_time = time.time()
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    print(f"Model Accuracy: 93.2% (validation)")
    
    # Load model
    print(f"\nLoading Stage2 model...")
    load_start = time.time()
    try:
        model = load_trained_model(model_path, num_classes=len(CLASS_NAMES), device=device)
        if model is None:
            print("Error: Failed to load model")
            return None
        load_time = time.time() - load_start
        print(f"OK: Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Load image
    print(f"\nLoading image...")
    try:
        img_pil = Image.open(image_path)
        img_np = np.array(img_pil.convert('L'))
        print(f"   Size: {img_pil.size}")
        print("OK: Image loaded")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Preprocess
    print(f"\nPreprocessing...")
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        print("OK: Image preprocessed to 224x224")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Make prediction
    print(f"\nMaking real-time prediction...")
    pred_start = time.time()
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        pred_time = time.time() - pred_start
        pred_class = CLASS_NAMES[pred_idx.item()]
        conf = confidence.item()
        prob_dict = {CLASS_NAMES[i]: float(probs[0][i].item()) for i in range(len(CLASS_NAMES))}
        
        print(f"OK: Prediction in {pred_time:.3f}s")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Display results
    print(f"\n{'='*70}")
    print("REAL-TIME PREDICTION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nPredicted Class: {pred_class}")
    print(f"Confidence: {conf:.2%}")
    
    print(f"\nCLASS PROBABILITIES:")
    print(f"{'Class':<15} {'Probability':<15} {'Confidence':<10}")
    print(f"{'-'*60}")
    for class_name in CLASS_NAMES:
        prob = prob_dict[class_name]
        bar_length = int(prob * 25)
        bar = "#" * bar_length + "-" * (25 - bar_length)
        marker = "*" if class_name == pred_class else " "
        print(f"{marker} {class_name:<13} {prob:>12.2%}  {bar}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"{'='*70}\n")
    
    return {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'predicted_class': pred_class,
        'confidence': conf,
        'probabilities': prob_dict,
        'model': os.path.basename(model_path),
        'model_accuracy': 0.932,
        'execution_time': total_time,
        'load_time': load_time,
        'prediction_time': pred_time
    }


def batch_real_time_predict(image_dir, model_path='models/stage2-best-epoch=19-val_acc=0.932.ckpt', device='cpu', max_images=None):
    """
    Real-time batch prediction
    """
    
    print("\n" + "="*70)
    print("REAL-TIME BATCH PREDICTION - STAGE2 MODEL")
    print("="*70)
    
    print(f"\nDirectory: {image_dir}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return None
    
    # Collect all images
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\nFound {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return None
    
    # Load model once
    print(f"\nLoading Stage2 model...")
    try:
        model = load_trained_model(model_path, num_classes=len(CLASS_NAMES), device=device)
        if model is None:
            return None
        print("OK: Model loaded")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Make predictions
    print(f"\nProcessing {len(image_files)} images...\n")
    results = []
    batch_start = time.time()
    
    for i, image_path in enumerate(image_files, 1):
        try:
            img = Image.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
            
            pred_class = CLASS_NAMES[pred_idx.item()]
            conf = confidence.item()
            
            results.append({
                'image': os.path.basename(image_path),
                'predicted_class': pred_class,
                'confidence': conf
            })
            
            status = "[OK]" if conf > 0.7 else "[LOW]"
            print(f"{i:3d}. {os.path.basename(image_path):<30} -> {pred_class:<12} {conf:>6.1%} {status}")
            
        except Exception as e:
            print(f"{i:3d}. {os.path.basename(image_path):<30} -> Error")
    
    batch_time = time.time() - batch_start
    
    print(f"\n{'='*70}")
    print(f"Batch Processing Complete")
    print(f"Total Images: {len(results)}")
    print(f"Total Time: {batch_time:.2f}s")
    print(f"Average per Image: {batch_time/len(results):.2f}s")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time prediction with Stage2 model")
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--batch', help='Path to image directory for batch processing')
    parser.add_argument('--model', default='models/stage2-best-epoch=19-val_acc=0.932.ckpt', help='Model path')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--max', type=int, help='Max images for batch processing')
    
    args = parser.parse_args()
    
    if args.image:
        result = real_time_predict(args.image, args.model, args.device)
        if result:
            print("SUCCESS: Prediction completed")
        else:
            print("FAILED: Could not complete prediction")
            sys.exit(1)
    
    elif args.batch:
        results = batch_real_time_predict(args.batch, args.model, args.device, args.max)
        if results:
            print(f"SUCCESS: {len(results)} images processed")
        else:
            print("FAILED: Could not process batch")
            sys.exit(1)
    
    else:
        parser.print_help()
