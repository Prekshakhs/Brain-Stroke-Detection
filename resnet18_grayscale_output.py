#!/usr/bin/env python3
"""
ResNet18 Grayscale Model - Output & Analysis Script
For Google Colab trained models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import argparse

CLASS_NAMES = ["Hemorrhagic", "Ischemic", "NoStroke"]


class ResNet18Grayscale(nn.Module):
    """ResNet18 adapted for grayscale images (1 input channel)"""
    
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        
        # Load ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Adapt first conv layer for grayscale (1 channel instead of 3)
        orig_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            1,  # Input channels: 1 for grayscale
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        
        # Average pretrained RGB weights if using pretrained
        if pretrained and hasattr(orig_conv, 'weight'):
            with torch.no_grad():
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                self.model.conv1.weight.copy_(mean_weight)
        
        # Replace final layer for 3 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def load_colab_model(model_path, device='cpu'):
    """Load model trained in Google Colab"""
    print(f"\n🔄 Loading Google Colab ResNet18 Grayscale model...")
    print(f"   Path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        # Create model architecture
        model = ResNet18Grayscale(num_classes=len(CLASS_NAMES), pretrained=False)
        
        # Load weights
        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location=device)
            
            # Try different loading strategies
            try:
                model.load_state_dict(state_dict, strict=False)
                print("✅ Loaded with direct method")
            except RuntimeError:
                try:
                    model.model.load_state_dict(state_dict, strict=False)
                    print("✅ Loaded into model.model wrapper")
                except RuntimeError:
                    try:
                        wrapped = {f'model.{k}': v for k, v in state_dict.items()}
                        model.load_state_dict(wrapped, strict=False)
                        print("✅ Loaded with wrapped keys")
                    except RuntimeError as e:
                        print(f"❌ Failed to load: {e}")
                        return None
        
        model.eval()
        model.to(device)
        print("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_output(model_path, test_data_dir='data/test', device='cpu'):
    """Get comprehensive output from ResNet18 Grayscale model"""
    
    print("\n" + "="*70)
    print("🧠 RESNET18 GRAYSCALE - MODEL OUTPUT ANALYSIS")
    print("="*70)
    
    print(f"\n📱 Device: {device}")
    print(f"📂 Model: {model_path}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model = load_colab_model(model_path, device=device)
    if model is None:
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
    
    # Preprocessing for grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Make predictions
    print(f"\n🔍 Making predictions...")
    predictions = []
    
    for i, image_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(image_paths)}")
        
        try:
            img = Image.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
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
                'hemorrhagic': prob_dict['Hemorrhagic'],
                'ischemic': prob_dict['Ischemic'],
                'nostroke': prob_dict['NoStroke'],
                'correct': pred_class == true_labels[i]
            })
            
        except Exception as e:
            print(f"   ⚠️ Error: {test_images[i]}")
    
    print(f"✅ Completed {len(predictions)} predictions")
    
    # Analyze results
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(predictions)
    correct = results_df['correct'].sum()
    total = len(results_df)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ OVERALL ACCURACY: {accuracy:.2%} ({correct}/{total})")
    
    print(f"\n📈 CONFIDENCE ANALYSIS:")
    print(f"   Mean: {results_df['confidence'].mean():.2%}")
    print(f"   Std: {results_df['confidence'].std():.2%}")
    print(f"   Min: {results_df['confidence'].min():.2%}")
    print(f"   Max: {results_df['confidence'].max():.2%}")
    
    print(f"\n📋 PER-CLASS ACCURACY:")
    print(f"{'Class':<15} {'Accuracy':<12} {'Images':<10}")
    print(f"{'-'*37}")
    for class_name in CLASS_NAMES:
        class_preds = results_df[results_df['true_class'] == class_name]
        if len(class_preds) > 0:
            class_acc = class_preds['correct'].sum() / len(class_preds)
            print(f"{class_name:<15} {class_acc:>10.2%}  {len(class_preds):>8}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['true_class'], results_df['predicted_class'], labels=CLASS_NAMES)
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"{'Predicted →':<15}", end='')
    for class_name in CLASS_NAMES:
        print(f"{class_name:<15}", end='')
    print()
    
    for i, true_class in enumerate(CLASS_NAMES):
        print(f"{true_class:<15}", end='')
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:<15}", end='')
        print()
    
    # Misclassifications
    wrong = results_df[results_df['correct'] == False]
    if len(wrong) > 0:
        print(f"\n⚠️ MISCLASSIFICATIONS ({len(wrong)}):")
        print(f"{'Image':<30} {'True':<15} {'Predicted':<15} {'Confidence':<12}")
        print(f"{'-'*72}")
        for idx, row in wrong.head(10).iterrows():
            print(f"{row['image']:<30} {row['true_class']:<15} {row['predicted_class']:<15} {row['confidence']:>10.2%}")
        if len(wrong) > 10:
            print(f"... and {len(wrong) - 10} more")
    else:
        print(f"\n✅ PERFECT - No misclassifications!")
    
    # Top correct predictions
    correct_preds = results_df[results_df['correct'] == True]
    if len(correct_preds) > 0:
        print(f"\n✅ TOP CORRECT PREDICTIONS:")
        top = correct_preds.nlargest(10, 'confidence')
        print(f"{'Image':<30} {'Class':<15} {'Confidence':<12}")
        print(f"{'-'*57}")
        for idx, row in top.iterrows():
            print(f"{row['image']:<30} {row['predicted_class']:<15} {row['confidence']:>10.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    output_file = f"results/resnet18_grayscale_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results: {output_file}")
    
    summary = {
        'model': model_path,
        'architecture': 'ResNet18 Grayscale',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'total_images': len(results_df),
        'accuracy': float(accuracy),
        'correct': int(correct),
        'incorrect': int(len(wrong)),
        'mean_confidence': float(results_df['confidence'].mean()),
        'per_class': {
            class_name: {
                'accuracy': float((results_df[results_df['true_class'] == class_name]['correct'].sum() / 
                                  len(results_df[results_df['true_class'] == class_name])) 
                                 if len(results_df[results_df['true_class'] == class_name]) > 0 else 0),
                'count': int(len(results_df[results_df['true_class'] == class_name]))
            }
            for class_name in CLASS_NAMES
        }
    }
    
    summary_file = f"results/resnet18_grayscale_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary: {summary_file}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 Grayscale Model Output Analysis")
    parser.add_argument('--model', required=True, help='Model path (.pth file)')
    parser.add_argument('--test-dir', default='data/test', help='Test data directory')
    parser.add_argument('--val-dir', default='data/Val', help='Validation directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Auto-detect test directory
    test_dir = args.test_dir if os.path.exists(args.test_dir) else args.val_dir
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                print(f"   - {f}")
        sys.exit(1)
    
    results = get_output(args.model, test_dir, device=args.device)
    
    if results is not None:
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")
