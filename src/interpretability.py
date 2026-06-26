# File: src/interpretability.py
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
    def _find_target_layer(self):
        """Find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:  # Conv layer has 4D output
                return layer.name
        return None
    
    def generate_heatmap(self, image, class_index):
        """Generate Grad-CAM heatmap"""
        if self.layer_name is None:
            print("No suitable layer found for Grad-CAM")
            return None
            
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_index]
        
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Compute guided gradients
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        
        # Compute weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
        # Compute weighted combination
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        # Normalize
        cam = tf.maximum(cam, 0)
        cam = cam / tf.reduce_max(cam) if tf.reduce_max(cam) > 0 else cam
        
        return cam.numpy()
    
    def visualize_prediction(self, image_path, predicted_class_idx, class_names):
        """Create Grad-CAM visualization"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(img_batch, predicted_class_idx)
        
        if heatmap is None:
            return None
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Superimpose heatmap on original image
        superimposed = (img_resized * 0.6 + heatmap_colored * 0.4).astype(np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(img_resized)
        axes[1].set_title('Preprocessed')
        axes[1].axis('off')
        
        axes[2].imshow(heatmap_resized, cmap='jet')
        axes[2].set_title('Grad-CAM Heatmap')
        axes[2].axis('off')
        
        axes[3].imshow(superimposed)
        axes[3].set_title('Superimposed')
        axes[3].axis('off')
        
        plt.suptitle(f'Prediction: {class_names[predicted_class_idx]}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return superimposed, heatmap_resized