#!/usr/bin/env python3
"""
Brain Stroke Detection - Model Comparison & Prediction Script
Use the pre-trained models in the models/ folder for predictions
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from model_def import ResNetClassifier

# Available models
MODELS = {
    'best_stroke': {
        'path': 'models/best_stroke.pth',
        'description': 'Best stroke detection model (main model)',
        'status': '✅ Primary Model'
    },
    'stage1': {
        'path': 'models/stage1-best-epoch=03-val_acc=0.638.ckpt',
        'description': 'Stage 1 model (63.8% validation accuracy)',
        'status': '⚠️ Experimental'
    },
    'stage2': {
        'path': 'models/stage2-best-epoch=19-val_acc=0.932.ckpt',
        'description': 'Stage 2 model (93.2% validation accuracy - Lightning format)',
        'status': '⭐ High Accuracy'
    }
}

CLASS_NAMES = ["Hemorrhagic", "Ischemic", "NoStroke"]


def list_available_models():
    """Display all available models"""
    print("\n" + "=" * 70)
    print("📊 AVAILABLE MODELS IN models/ FOLDER")
    print("=" * 70)
    
    for name, info in MODELS.items():
        path = info['path']
        status = info['status']
        desc = info['description']
        exists = "✅" if os.path.exists(path) else "❌"
        
        print(f"\n{exists} {name.upper()}")
        print(f"   Description: {desc}")
        print(f"   Path: {path}")
        print(f"   Status: {status}")
        
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   Size: {size_mb:.1f} MB")


def load_model(model_name='best_stroke', device='cpu'):
    """Load model from models/ folder"""
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return None
    
    model_path = MODELS[model_name]['path']
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n🔄 Loading {model_name} model...")
    print(f"   Path: {model_path}")
    
    try:
        model = ResNetClassifier(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different state dict formats
        if 'model.conv1.weight' not in state_dict and 'conv1.weight' in state_dict:
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image_path):
    """Convert image to model input"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(model_name, image_path, show_all_probs=True):
    """Make prediction using specified model"""
    
    # Validate inputs
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(model_name, device=device)
    if model is None:
        return None
    
    # Preprocess image
    print(f"\n📸 Processing image: {os.path.basename(image_path)}")
    input_tensor = preprocess_image(image_path).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    
    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence = conf.item()
    
    # Results
    print("\n" + "=" * 70)
    print(f"🎯 PREDICTION RESULTS ({model_name})")
    print("=" * 70)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    if show_all_probs:
        print(f"\nClass Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = probs[i].item()
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {class_name:15} {bar} {prob:.2%}")
    
    print("=" * 70)
    
    return {
        "model": model_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))}
    }


def predict_batch(model_name, image_dir):
    """Make predictions on all images in a directory"""
    print(f"\n📁 Predicting on images in: {image_dir}")
    
    if not os.path.isdir(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name, device=device)
    
    if model is None:
        return
    
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("=" * 70)
    
    results = []
    for image_path in image_files[:10]:  # Limit to first 10 for demo
        try:
            input_tensor = preprocess_image(str(image_path)).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
            
            pred_class = CLASS_NAMES[pred_idx.item()]
            results.append({
                'file': image_path.name,
                'predicted': pred_class,
                'confidence': conf.item()
            })
            
            print(f"✅ {image_path.name:30} → {pred_class:15} ({conf:.2%})")
        except Exception as e:
            print(f"❌ {image_path.name:30} → Error: {e}")
    
    print("=" * 70)
    print(f"Processed {len(results)} images")
    
    return results


def interactive_prediction():
    """Interactive mode for predictions"""
    print("\n" + "=" * 70)
    print("🧠 BRAIN STROKE DETECTION - INTERACTIVE PREDICTION")
    print("=" * 70)
    
    # Show available models
    list_available_models()
    
    # Select model
    print("\n" + "-" * 70)
    print("SELECT MODEL:")
    for i, model_name in enumerate(MODELS.keys(), 1):
        print(f"{i}. {model_name} - {MODELS[model_name]['description']}")
    
    choice = input("\nEnter model number (1-3): ").strip()
    
    try:
        model_idx = int(choice) - 1
        model_names = list(MODELS.keys())
        if 0 <= model_idx < len(model_names):
            selected_model = model_names[model_idx]
        else:
            print("❌ Invalid choice")
            return
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Get image path
    print("\n" + "-" * 70)
    image_path = input("Enter image path: ").strip()
    
    if not image_path:
        print("❌ No image path provided")
        return
    
    # Make prediction
    result = predict_image(selected_model, image_path, show_all_probs=True)
    
    if result:
        print(f"\n✅ Prediction saved!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain Stroke Detection - Prediction using available models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_all_models.py --list
  python predict_all_models.py --model best_stroke --image "data/test/Ischemic/sample.jpg"
  python predict_all_models.py --model stage2 --image "data/test/NoStroke/sample.jpg"
  python predict_all_models.py --batch "data/test/Hemorrhagic" --model best_stroke
  python predict_all_models.py --interactive
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--model', type=str, default='best_stroke', 
                       help='Model to use (best_stroke, stage1, stage2)')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--batch', type=str, help='Directory containing images to predict')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--no-probs', action='store_true', help='Don\'t show all probabilities')
    
    args = parser.parse_args()
    
    # List models
    if args.list or (not args.image and not args.batch and not args.interactive):
        list_available_models()
        if not args.image and not args.batch and not args.interactive:
            print("\n💡 Try: python predict_all_models.py --model best_stroke --image <path>")
            print("   Or: python predict_all_models.py --interactive")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_prediction()
        return
    
    # Single image prediction
    if args.image:
        predict_image(args.model, args.image, show_all_probs=not args.no_probs)
        return
    
    # Batch prediction
    if args.batch:
        predict_batch(args.model, args.batch)
        return
    
    print("❌ No action specified. Use --help for options.")


if __name__ == "__main__":
    main()
