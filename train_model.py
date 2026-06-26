# File: train_model.py
"""
Brain Stroke Detection - Training Script
Run this script to train the model on your dataset
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from stroke_detection_model import StrokeDetectionCNN
from data_utils import DatasetManager

def main():
    parser = argparse.ArgumentParser(description='Train Brain Stroke Detection Model')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'custom'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (will be img_size x img_size)')
    
    args = parser.parse_args()
    
    print("🧠 BRAIN STROKE DETECTION - TRAINING")
    print("=" * 40)
    print(f"📊 Configuration:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Model type: {args.model_type}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Image size: {args.img_size}x{args.img_size}")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    # Verify dataset
    dataset_manager = DatasetManager(args.data_dir)
    if not dataset_manager.verify_dataset():
        print("❌ Dataset verification failed!")
        return
    
    # Create model
    detector = StrokeDetectionCNN(
        img_size=(args.img_size, args.img_size),
        num_classes=3
    )
    
    # Create and compile model
    model = detector.create_model(model_type=args.model_type)
    detector.compile_model(learning_rate=args.learning_rate)
    
    print(f"\n📊 Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()
    
    # Create data generators
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    train_gen, val_gen, test_gen = detector.create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size
    )
    
    if train_gen.samples == 0:
        print("❌ No training samples found!")
        return
    
    if val_gen.samples == 0:
        print("❌ No validation samples found!")
        return
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = detector.train_model(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=args.epochs
    )
    
    if history is None:
        print("❌ Training failed!")
        return
    
    # Save model
    model_path = f'models/stroke_model_{args.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    detector.save_model(model_path)
    
    # Evaluate on test set if available
    if test_gen and test_gen.samples > 0:
        print(f"\n📊 Evaluating on test set...")
        results = detector.evaluate_model(test_gen)
        print(f"✅ Test Accuracy: {results['accuracy']:.4f}")
    else:
        print("⚠️ No test set available for evaluation")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print(f"📁 Results saved to: results/")

if __name__ == "__main__":
    main()