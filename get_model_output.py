#!/usr/bin/env python3
"""
Get Comprehensive Model Output - Predictions with Full Analysis
Works with Google Colab trained models or local trained models
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from model_def import ResNetClassifier, load_trained_model
from datetime import datetime
import json

CLASS_NAMES = ["Hemorrhagic", "Ischemic", "NoStroke"]


def get_comprehensive_output(model_path, test_data_dir='data/test', output_format='full'):
    """
    Get comprehensive output from model predictions
    
    Args:
        model_path: Path to trained model (.pth or .ckpt)
        test_data_dir: Directory with test images organized by class
        output_format: 'full', 'summary', or 'detailed'
    """
    
    print("\n" + "="*70)
    print("🧠 BRAIN STROKE DETECTION - MODEL OUTPUT ANALYSIS")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    print(f"📂 Model: {model_path}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for f in os.listdir('models'):
            if f.endswith(('.pth', '.ckpt')):
                print(f"   - {f}")
        return None
    
    # Load model
    print(f"\n🔄 Loading model...")
    try:
        model = load_trained_model(model_path, num_classes=len(CLASS_NAMES), device=device)
        if model is None:
            print("❌ Failed to load model")
            return None
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load test images
    print(f"\n📁 Loading test images from {test_data_dir}...")
    
    test_images = []
    image_paths = []
    true_labels = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_data_dir, class_name)
        if os.path.exists(class_dir):
            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_images.append(img_file)
                    image_paths.append(os.path.join(class_dir, img_file))
                    true_labels.append(class_name)
    
    print(f"✅ Loaded {len(test_images)} test images")
    print(f"   - Hemorrhagic: {true_labels.count('Hemorrhagic')}")
    print(f"   - Ischemic: {true_labels.count('Ischemic')}")
    print(f"   - NoStroke: {true_labels.count('NoStroke')}")
    
    # Make predictions
    print(f"\n🔍 Making predictions...")
    predictions = []
    results = []
    
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(image_paths)}")
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            img = Image.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
            
            pred_class = CLASS_NAMES[pred_idx.item()]
            conf = confidence.item()
            prob_dict = {CLASS_NAMES[j]: float(probs[0][j].item()) for j in range(len(CLASS_NAMES))}
            
            predictions.append({
                'image': test_images[i],
                'true_class': true_labels[i],
                'predicted_class': pred_class,
                'confidence': conf,
                'probabilities': prob_dict,
                'correct': pred_class == true_labels[i]
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing {test_images[i]}: {str(e)}")
            continue
    
    print(f"✅ Completed predictions on {len(predictions)} images")
    
    # Calculate metrics
    print(f"\n{'='*70}")
    print("📊 MODEL OUTPUT ANALYSIS")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(predictions)
    correct = results_df['correct'].sum()
    total = len(results_df)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ OVERALL ACCURACY: {accuracy:.2%} ({correct}/{total})")
    print(f"\n📈 CONFIDENCE STATISTICS:")
    print(f"   Mean Confidence: {results_df['confidence'].mean():.2%}")
    print(f"   Std Dev: {results_df['confidence'].std():.2%}")
    print(f"   Min Confidence: {results_df['confidence'].min():.2%}")
    print(f"   Max Confidence: {results_df['confidence'].max():.2%}")
    
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<15} {'Accuracy':<12} {'Images':<10}")
    print(f"{'-'*37}")
    for class_name in CLASS_NAMES:
        class_preds = results_df[results_df['true_class'] == class_name]
        if len(class_preds) > 0:
            class_acc = class_preds['correct'].sum() / len(class_preds)
            print(f"{class_name:<15} {class_acc:>10.2%}  {len(class_preds):>8}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"{'Predicted →':<15}", end='')
    for class_name in CLASS_NAMES:
        print(f"{class_name:<15}", end='')
    print()
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['true_class'], results_df['predicted_class'], labels=CLASS_NAMES)
    
    for i, true_class in enumerate(CLASS_NAMES):
        print(f"{true_class:<15}", end='')
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:<15}", end='')
        print()
    
    # Wrong predictions
    wrong_preds = results_df[results_df['correct'] == False]
    if len(wrong_preds) > 0:
        print(f"\n⚠️ MISCLASSIFICATIONS ({len(wrong_preds)}):")
        print(f"{'Image':<30} {'True':<15} {'Predicted':<15} {'Confidence':<12}")
        print(f"{'-'*72}")
        for idx, row in wrong_preds.head(10).iterrows():
            print(f"{row['image']:<30} {row['true_class']:<15} {row['predicted_class']:<15} {row['confidence']:>10.2%}")
        if len(wrong_preds) > 10:
            print(f"... and {len(wrong_preds) - 10} more")
    else:
        print(f"\n✅ PERFECT ACCURACY - No misclassifications!")
    
    # High confidence correct predictions
    correct_preds = results_df[results_df['correct'] == True]
    if len(correct_preds) > 0:
        top_correct = correct_preds.nlargest(10, 'confidence')
        print(f"\n✅ TOP CORRECT PREDICTIONS (High Confidence):")
        print(f"{'Image':<30} {'Class':<15} {'Confidence':<12}")
        print(f"{'-'*57}")
        for idx, row in top_correct.iterrows():
            print(f"{row['image']:<30} {row['predicted_class']:<15} {row['confidence']:>10.2%}")
    
    # Save detailed results
    output_file = f"results/model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save summary
    summary = {
        'model': model_path,
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'total_images': len(results_df),
        'accuracy': float(accuracy),
        'correct_predictions': int(correct),
        'misclassifications': int(len(wrong_preds)),
        'mean_confidence': float(results_df['confidence'].mean()),
        'per_class_accuracy': {
            class_name: float((results_df[results_df['true_class'] == class_name]['correct'].sum() / 
                             len(results_df[results_df['true_class'] == class_name])) 
                            if len(results_df[results_df['true_class'] == class_name]) > 0 else 0)
            for class_name in CLASS_NAMES
        }
    }
    
    summary_file = f"results/model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to: {summary_file}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get comprehensive model output")
    parser.add_argument('--model', default='models/best_stroke.pth', help='Model path')
    parser.add_argument('--test-dir', default='data/test', help='Test data directory')
    parser.add_argument('--val-dir', default='data/Val', help='Validation data directory (alternative)')
    
    args = parser.parse_args()
    
    # Try test directory first, then validation
    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.val_dir
    
    results = get_comprehensive_output(args.model, test_dir)
    
    if results is not None:
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")
