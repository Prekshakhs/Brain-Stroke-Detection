# File: src/model_ensemble.py
import numpy as np
import tensorflow as tf
from stroke_detection_model import StrokeDetectionCNN

class ModelEnsemble:
    def __init__(self, model_paths, weights=None):
        self.model_paths = model_paths
        self.models = []
        self.weights = weights or [1.0] * len(model_paths)
        self.class_names = ['No_Stroke', 'Hemorrhagic_Stroke', 'Ischemic_Stroke']
        
        # Load all models
        for path in model_paths:
            detector = StrokeDetectionCNN()
            if detector.load_model(path):
                self.models.append(detector)
            else:
                print(f"Failed to load model: {path}")
    
    def predict_ensemble(self, image_path):
        """Make ensemble prediction"""
        if not self.models:
            print("No models loaded for ensemble prediction")
            return None
        
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            result = model.predict_single_image(image_path)
            if result:
                pred_probs = [result['probabilities'][class_name] for class_name in self.class_names]
                predictions.append(pred_probs)
        
        if not predictions:
            return None
        
        # Weighted average of predictions
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
        
        # Get final prediction
        predicted_class_idx = np.argmax(weighted_predictions)
        confidence = weighted_predictions[predicted_class_idx]
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(weighted_predictions[i])
                for i in range(len(self.class_names))
            },
            'individual_predictions': predictions
        }
        
        return result