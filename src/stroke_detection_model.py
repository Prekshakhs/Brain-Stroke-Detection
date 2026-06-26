import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrokeDetectionCNN:
    def __init__(self, img_size=(224, 224), num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['NoStroke', 'Hemorrhagic', 'Ischemic']
        self.model = None
        self.history = None
        
    def create_model(self, model_type='efficientnet', fine_tune=False):
        """Create CNN model with transfer learning"""
        print(f"Creating {model_type} model...")
        
        if model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif model_type == 'resnet':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Input(shape=(*self.img_size, 3)),
                
                # First conv block
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second conv block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third conv block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fourth conv block
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                
                # Classifier
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            print("✅ Custom CNN model created!")
            return self.model
        
        # For transfer learning models
        base_model.trainable = fine_tune
        
        # Add custom classification layers
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        print(f"✅ {model_type} model created!")
        return self.model
    
    def compile_model(self, learning_rate=0.001, use_all_metrics=False):
        """Compile model with optimizer and loss function - FIXED VERSION"""
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        if use_all_metrics:
            # Use explicit metric objects (can cause issues in some TF versions)
            try:
                metrics = [
                    CategoricalAccuracy(name='accuracy'),
                    Precision(name='precision', average='macro'),
                    Recall(name='recall', average='macro')
                ]
            except:
                # Fallback to simple metrics
                metrics = ['accuracy']
                print("⚠️ Using simplified metrics due to compatibility issues")
        else:
            # Safe option - only use accuracy
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        print("✅ Model compiled successfully!")
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, test_dir=None, batch_size=16):
        """Create data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation and test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        test_generator = None
        if test_dir and os.path.exists(test_dir):
            test_generator = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='categorical',
                classes=self.class_names,
                shuffle=False
            )
        
        print(f"✅ Data generators created:")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        if test_generator:
            print(f"   Test samples: {test_generator.samples}")
        
        return train_generator, val_generator, test_generator
    
    def create_callbacks(self, model_save_path='models/best_stroke_model.h5'):
        """Create training callbacks"""
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Only add CSV logger if results directory can be created
        try:
            os.makedirs('results', exist_ok=True)
            callbacks.append(keras.callbacks.CSVLogger('results/training_log.csv', append=True))
        except:
            print("⚠️ Could not create results directory for CSV logging")
        
        return callbacks
    
    def train_model(self, train_generator, val_generator, epochs=50, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = self.create_callbacks()
        
        print("🚀 Starting training...")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Epochs: {epochs}")
        
        try:
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            print("✅ Training completed!")
            return self.history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Try reducing batch size or using simpler metrics")
            return None
    
    def evaluate_model(self, test_generator):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model...")
        
        try:
            # Get predictions
            test_generator.reset()
            predictions = self.model.predict(test_generator, verbose=1)
            y_pred = np.argmax(predictions, axis=1)
            y_true = test_generator.classes
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_true)
            
            # Classification report
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # AUC score (multiclass) - with error handling
            try:
                y_true_binary = label_binarize(y_true, classes=[0, 1, 2])
                auc_score = roc_auc_score(y_true_binary, predictions, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"⚠️ Could not calculate AUC score: {e}")
                auc_score = "Unable to calculate"
            
            # Print results
            print(f"\n📊 Model Evaluation Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score}")
            print(f"\n📋 Classification Report:")
            print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
            
            # Save results
            results = {
                'accuracy': float(accuracy),
                'auc_score': auc_score if isinstance(auc_score, str) else float(auc_score),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results to file
            try:
                os.makedirs('results', exist_ok=True)
                with open('results/evaluation_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print("✅ Results saved to results/evaluation_results.json")
            except Exception as e:
                print(f"⚠️ Could not save results: {e}")
            
            # Plot confusion matrix
            try:
                self.plot_confusion_matrix(cm)
            except Exception as e:
                print(f"⚠️ Could not plot confusion matrix: {e}")
            
            # Plot training history if available
            if self.history:
                try:
                    self.plot_training_history()
                except Exception as e:
                    print(f"⚠️ Could not plot training history: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        try:
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("✅ Confusion matrix saved to results/confusion_matrix.png")
        except:
            print("⚠️ Could not save confusion matrix plot")
        
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("No training history available")
            return
        
        # Get available metrics
        available_metrics = list(self.history.history.keys())
        
        # Determine subplot configuration based on available metrics
        if 'accuracy' in available_metrics and 'val_accuracy' in available_metrics:
            if any(metric in available_metrics for metric in ['precision', 'recall']):
                # Full 2x2 plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
            else:
                # Only accuracy and loss
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            # Minimal plot
            fig, axes = plt.subplots(1, 1, figsize=(8, 5))
            axes = [axes]
        
        plot_idx = 0
        
        # Accuracy plot
        if 'accuracy' in available_metrics and 'val_accuracy' in available_metrics:
            axes[plot_idx].plot(self.history.history['accuracy'], 'b-', label='Training', linewidth=2)
            axes[plot_idx].plot(self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
            axes[plot_idx].set_title('Model Accuracy', fontsize=14)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Loss plot
        if 'loss' in available_metrics and 'val_loss' in available_metrics and plot_idx < len(axes):
            axes[plot_idx].plot(self.history.history['loss'], 'b-', label='Training', linewidth=2)
            axes[plot_idx].plot(self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
            axes[plot_idx].set_title('Model Loss', fontsize=14)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Loss')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Precision plot (if available)
        if 'precision' in available_metrics and 'val_precision' in available_metrics and plot_idx < len(axes):
            axes[plot_idx].plot(self.history.history['precision'], 'b-', label='Training', linewidth=2)
            axes[plot_idx].plot(self.history.history['val_precision'], 'r-', label='Validation', linewidth=2)
            axes[plot_idx].set_title('Model Precision', fontsize=14)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Precision')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Recall plot (if available)
        if 'recall' in available_metrics and 'val_recall' in available_metrics and plot_idx < len(axes):
            axes[plot_idx].plot(self.history.history['recall'], 'b-', label='Training', linewidth=2)
            axes[plot_idx].plot(self.history.history['val_recall'], 'r-', label='Validation', linewidth=2)
            axes[plot_idx].set_title('Model Recall', fontsize=14)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Recall')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
            print("✅ Training history saved to results/training_history.png")
        except:
            print("⚠️ Could not save training history plot")
        
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict stroke type for a single image"""
        try:
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(prediction[0][i]) 
                    for i in range(len(self.class_names))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None
    
    def save_model(self, filepath='models/final_stroke_detection_model.h5'):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if self.model:
                self.model.save(filepath)
                print(f"✅ Model saved to {filepath}")
                
                # Save model architecture as JSON
                try:
                    model_json = self.model.to_json()
                    json_path = filepath.replace('.h5', '_architecture.json')
                    with open(json_path, 'w') as f:
                        f.write(model_json)
                    print(f"✅ Model architecture saved to {json_path}")
                except Exception as e:
                    print(f"⚠️ Could not save model architecture: {e}")
            else:
                print("❌ No model to save!")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Brain Stroke Detection CNN - Model Implementation")
    print("=" * 55)
    
    # Create model instance
    detector = StrokeDetectionCNN(img_size=(224, 224), num_classes=3)
    
    # Create and compile model (using simple metrics to avoid errors)
    model = detector.create_model(model_type='efficientnet')
    detector.compile_model(learning_rate=0.001, use_all_metrics=False)
    
    # Print model summary
    print(f"\n📊 Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    
    print("\n✅ Model implementation ready!")
    print("📋 Next steps:")
    print("1. Ensure your dataset is properly organized")
    print("2. Run the training script")
    print("3. Evaluate the model")

# IMMEDIATE SOLUTION - Run this to fix your current training:

print("""
🔧 IMMEDIATE FIX FOR YOUR CURRENT ERROR:

1. Stop your current training (Ctrl+C)

2. Replace your src/stroke_detection_model.py with the corrected version above

3. Restart training with simplified metrics:
   python train_model.py --epochs 30 --batch_size 16

4. If still having issues, try with smaller batch size:
   python train_model.py --epochs 30 --batch_size 8

The main fix is using only 'accuracy' metric instead of string-based 'precision' and 'recall' 
which were causing the TypeError.
""")