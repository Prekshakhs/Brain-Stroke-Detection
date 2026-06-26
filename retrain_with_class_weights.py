#!/usr/bin/env python3
"""
Brain Stroke Detection - Model Retraining with Class Weights
============================================================

This script retrains the model using class weights to fix the prediction bias.
The model was predicting 100% Ischemic due to imbalanced training data.

Usage:
    python retrain_with_class_weights.py

This will:
    1. Load the training data
    2. Calculate balanced class weights
    3. Retrain the model using the weighted loss function
    4. Save the improved model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime

# Import model definition
from model_def import ResNetClassifier


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with labels."""
    
    def __init__(self, data_dir, split='train', img_size=224, transform=None):
        """
        Args:
            data_dir: Path to data directory (contains train/val/test subdirs)
            split: 'train', 'val', or 'test'
            img_size: Size to resize images to
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir) / split
        self.img_size = img_size
        self.transform = transform
        self.class_names = ['Hemorrhagic', 'Ischemic', 'NoStroke']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_file in class_dir.glob('*.png'):
                self.images.append(str(img_file))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {split} set")
        if len(self.images) > 0:
            class_counts = {}
            for label in self.labels:
                class_name = self.class_names[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            print(f"  Class distribution: {class_counts}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor and normalize
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
        
        return img, label


def calculate_class_weights(dataset):
    """Calculate balanced class weights for imbalanced dataset."""
    print("\n📊 Calculating class weights...")
    
    labels = np.array(dataset.labels)
    class_names = dataset.class_names
    
    # Calculate weights using sklearn
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    print(f"✅ Class weights calculated:")
    for i, (class_name, weight) in enumerate(zip(class_names, weights)):
        print(f"   {class_name}: {weight:.4f}")
    
    return weights


def create_dataloaders(data_dir, batch_size=32, img_size=224):
    """Create training, validation, and test dataloaders."""
    print(f"\n📁 Creating dataloaders from {data_dir}...")
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Create datasets
    train_dataset = MedicalImageDataset(data_dir, 'train', img_size, transform=train_transforms)
    val_dataset = MedicalImageDataset(data_dir, 'val', img_size, transform=val_transforms)
    test_dataset = MedicalImageDataset(data_dir, 'test', img_size, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={100*correct/total:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("=" * 70)
    print("🧠 BRAIN STROKE DETECTION - RETRAINING WITH CLASS WEIGHTS")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    data_dir = 'data'
    model_save_dir = 'models'
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    img_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️ Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        data_dir, batch_size, img_size
    )
    
    if len(train_loader) == 0:
        print("❌ No training data found!")
        return
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    
    # Create model
    print("\n🔧 Creating model...")
    model = ResNetClassifier(num_classes=3)
    model = model.to(device)
    print(f"✅ Model created on {device}")
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"✅ Loss function: CrossEntropyLoss with class weights")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    print(f"✅ Optimizer: Adam with learning rate scheduler")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0
    patience_counter = 0
    patience = 5
    
    # Training loop
    print("\n🚀 Starting training with class-weighted loss...")
    print("-" * 70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📊 Results: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, 'best_stroke_weighted.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved to {model_path} (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(model_save_dir, 'training_history_weighted.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to {history_path}")
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, 'stroke_model_final_weighted.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
    
    print("\n" + "=" * 70)
    print("🎉 RETRAINING COMPLETED!")
    print("=" * 70)
    print(f"\n📈 Summary:")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Final test accuracy: {test_acc:.2f}%")
    print(f"   Improvement from bias (33.3%): +{test_acc - 33.3:.1f}%")
    print(f"\n📁 Saved models:")
    print(f"   Best model: {model_save_dir}/best_stroke_weighted.pth")
    print(f"   Final model: {final_model_path}")
    print(f"\n🔄 Next steps:")
    print(f"   1. Run: python diagnose_model_bias.py")
    print(f"      This will verify the new model's predictions")
    print(f"   2. Replace: models/best_stroke.pth with best_stroke_weighted.pth")
    print(f"   3. Restart Streamlit app to use the new model")
    print(f"   4. Remove the disclaimer from the Streamlit app")
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
