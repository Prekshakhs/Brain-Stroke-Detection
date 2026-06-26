"""
model_def.py
Defines the ResNetClassifier architecture for Brain Stroke Detection.
✅ Compatible with both .pth (state_dict) and .ckpt (Lightning checkpoint) files.
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-4, pretrained=False, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()

        # --- Load ResNet18 backbone ---
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # --- Adapt first conv layer for grayscale images (1 channel) ---
        orig_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            1,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )

        # If pretrained, average RGB weights to initialize grayscale conv
        if pretrained and hasattr(orig_conv, 'weight'):
            with torch.no_grad():
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                self.model.conv1.weight.copy_(mean_weight)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Hyperparameters
        self.lr = lr

    # --- forward pass ---
    def forward(self, x):
        return self.model(x)

    # --- training step ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    # --- validation step ---
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    # --- optimizer ---
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}}


# -------------------------------------------------------------------
# Helper: load model automatically from either .pth or .ckpt
# -------------------------------------------------------------------
def load_trained_model(model_path, num_classes=3, device=None):
    """
    Load a trained model for inference.

    Args:
        model_path (str): Path to .pth or .ckpt file
        num_classes (int): Number of output classes
        device (torch.device or str): 'cuda' or 'cpu'
    Returns:
        model (ResNetClassifier): Loaded and ready-to-use model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ResNetClassifier(num_classes=num_classes)

    try:
        if model_path.endswith(".pth"):
            print(f"Loading state_dict model: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            
            # Strategy 1: Try direct load
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Model loaded with direct method")
            except RuntimeError as e:
                print(f"Direct loading failed: {str(e)[:100]}...")
                print("Attempting alternative loading method...")
                
                # Strategy 2: Try loading into model.model wrapper
                try:
                    model.model.load_state_dict(state_dict, strict=False)
                    print("Model loaded into model.model wrapper")
                except RuntimeError:
                    # Strategy 3: Try wrapping keys with "model." prefix
                    try:
                        wrapped_state_dict = {"model." + k: v for k, v in state_dict.items()}
                        model.load_state_dict(wrapped_state_dict, strict=False)
                        print("Model loaded with wrapped keys")
                    except RuntimeError:
                        # Strategy 4: Try removing "model." prefix
                        try:
                            unwrapped_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                            model.model.load_state_dict(unwrapped_state_dict, strict=False)
                            print("Model loaded with unwrapped keys")
                        except RuntimeError as final_error:
                            print(f"All loading strategies failed!")
                            raise final_error
                    
        elif model_path.endswith(".ckpt"):
            print(f"Loading Lightning checkpoint: {model_path}")
            model = ResNetClassifier.load_from_checkpoint(model_path, map_location=device)
        else:
            raise ValueError("Unsupported model format! Use a .pth or .ckpt file")

        model.eval()
        model.to(device)
        print("Model loaded successfully and ready for inference.")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"   Model path: {model_path}")
        print(f"   File exists: {os.path.exists(model_path)}")
        raise
