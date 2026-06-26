#!/usr/bin/env python
"""
Test script to verify the entire prediction pipeline works
"""
import torch
import sys
import os
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_def import load_trained_model

def test_model_loading():
    """Test if model loads correctly"""
    print("=" * 60)
    print("🧪 Testing Model Loading")
    print("=" * 60)
    
    model_path = "models/best_stroke.pth"
    print(f"\n📂 Model path: {model_path}")
    print(f"✅ File exists: {os.path.exists(model_path)}")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {device}")
        
        model = load_trained_model(model_path, num_classes=3, device=device)
        print(f"✅ Model loaded successfully!")
        print(f"✅ Model is in eval mode: {not model.training}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_preprocessing(image_path):
    """Test image preprocessing"""
    print("\n" + "=" * 60)
    print("🧪 Testing Image Preprocessing")
    print("=" * 60)
    
    print(f"\n📸 Image path: {image_path}")
    
    try:
        img = Image.open(image_path)
        print(f"✅ Image loaded: {img.size}")
        
        # Convert to grayscale
        img_gray = img.convert('L')
        print(f"✅ Converted to grayscale")
        
        # Resize
        img_resized = img_gray.resize((224, 224))
        print(f"✅ Resized to 224x224")
        
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0)
        print(f"✅ Converted to tensor: {img_tensor.shape}")
        
        return img_tensor
    except Exception as e:
        print(f"❌ Failed to preprocess image: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_prediction(model, image_tensor):
    """Test prediction"""
    print("\n" + "=" * 60)
    print("🧪 Testing Prediction")
    print("=" * 60)
    
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_names = ['Hemorrhagic', 'Ischemic', 'NoStroke']
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        print(f"\n✅ Prediction successful!")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence_score:.2%}")
        print(f"\n   Class probabilities:")
        for i, class_name in enumerate(class_names):
            prob = probabilities[0][i].item()
            print(f"      {class_name}: {prob:.2%}")
        
        return predicted_class, confidence_score
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("BRAIN STROKE DETECTION - PIPELINE TEST")
    print("🚀" * 30)
    
    # Test 1: Load model
    model = test_model_loading()
    if model is None:
        print("\n❌ Cannot proceed - model loading failed")
        return
    
    # Test 2: Find a test image
    test_image_dir = "data/test/Hemorrhagic"
    if os.path.exists(test_image_dir):
        images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            test_image = os.path.join(test_image_dir, images[0])
            print(f"\n📸 Using test image: {test_image}")
            
            # Test 3: Preprocess image
            img_tensor = test_image_preprocessing(test_image)
            if img_tensor is None:
                print("\n❌ Cannot proceed - image preprocessing failed")
                return
            
            # Test 4: Make prediction
            predicted_class, confidence = test_prediction(model, img_tensor)
            if predicted_class is None:
                print("\n❌ Cannot proceed - prediction failed")
                return
            
            print("\n" + "✅" * 30)
            print("ALL TESTS PASSED!")
            print("✅" * 30)
            print("\n🎉 Your model is ready for use in Streamlit!")
            print("🎉 Run: python run_streamlit.py")
        else:
            print(f"\n⚠️ No images found in {test_image_dir}")
    else:
        print(f"\n⚠️ Test image directory not found: {test_image_dir}")

if __name__ == "__main__":
    main()
