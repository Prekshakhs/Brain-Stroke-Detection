"""
diagnose_model_bias.py

Comprehensive diagnostic script to identify why the model predicts "No Stroke" for all images.
This script analyzes:
1. Model output distributions
2. Training data balance
3. Model confidence patterns
4. Grad-CAM focus areas
5. Per-class prediction statistics
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_def import load_trained_model, ResNetClassifier
from torchvision import transforms

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_stroke.pth"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['Hemorrhagic', 'Ischemic', 'NoStroke']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔍 DIAGNOSING MODEL BIAS")
print(f"=" * 70)
print(f"📱 Device: {DEVICE}")
print(f"🧠 Model: {MODEL_PATH}")
print(f"📂 Test Data: {TEST_DATA_DIR}")
print()

# ============================================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================================

print("Step 1: Loading model...")
try:
    model = load_trained_model(str(MODEL_PATH), num_classes=3, device=DEVICE)
    print("✅ Model loaded successfully\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: COLLECT ALL TEST DATA
# ============================================================================

print("Step 2: Collecting test data...")

test_images = {}
image_paths = {}

for class_name in CLASS_NAMES:
    class_dir = TEST_DATA_DIR / class_name
    if class_dir.exists():
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        test_images[class_name] = len(images)
        image_paths[class_name] = images
        print(f"  {class_name}: {len(images)} images")
    else:
        print(f"  ⚠️  {class_name}: NOT FOUND")
        test_images[class_name] = 0
        image_paths[class_name] = []

total_images = sum(test_images.values())
print(f"\n📊 Total test images: {total_images}\n")

# ============================================================================
# STEP 3: RUN PREDICTIONS AND COLLECT STATISTICS
# ============================================================================

print("Step 3: Running predictions on all test images...")

def preprocess_image(image_path):
    """Preprocess image for model"""
    img = Image.open(image_path).convert('L')
    img = img.resize((224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(img).unsqueeze(0)

predictions_by_class = {}
all_raw_outputs = []
all_predictions = []

model.eval()

for true_class in CLASS_NAMES:
    predictions_by_class[true_class] = {
        'Hemorrhagic': 0,
        'Ischemic': 0,
        'NoStroke': 0,
        'total': 0,
        'confidences': [],
        'raw_outputs': []
    }

print()
for true_class in CLASS_NAMES:
    class_images = image_paths[true_class]
    
    print(f"Processing {true_class}... ({len(class_images)} images)")
    
    for img_idx, img_path in enumerate(class_images):
        try:
            # Preprocess
            img_tensor = preprocess_image(img_path).to(DEVICE)
            
            # Get raw model output (logits)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
            
            # Get prediction
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = probs[0, pred_class_idx].item()
            
            # Store statistics
            predictions_by_class[true_class]['total'] += 1
            predictions_by_class[true_class][pred_class] += 1
            predictions_by_class[true_class]['confidences'].append(confidence)
            
            # Store raw outputs for analysis
            all_raw_outputs.append({
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'logits': logits[0].cpu().numpy(),
                'probabilities': probs[0].cpu().numpy(),
                'image': img_path.name
            })
            
            if (img_idx + 1) % 50 == 0:
                print(f"  Processed {img_idx + 1}/{len(class_images)}")
        
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")

print("\n")

# ============================================================================
# STEP 4: ANALYZE RESULTS
# ============================================================================

print("Step 4: Analyzing prediction patterns...\n")

# Print confusion matrix
print("📊 CONFUSION MATRIX (Predicted vs Actual)")
print("-" * 70)
print(f"{'True Class':<20} {'→ Hemorrhagic':<15} {'→ Ischemic':<15} {'→ NoStroke':<15}")
print("-" * 70)

for true_class in CLASS_NAMES:
    stats = predictions_by_class[true_class]
    total = stats['total']
    
    if total > 0:
        h_pct = (stats['Hemorrhagic'] / total) * 100
        i_pct = (stats['Ischemic'] / total) * 100
        n_pct = (stats['NoStroke'] / total) * 100
        
        print(f"{true_class:<20} {h_pct:>5.1f}% ({stats['Hemorrhagic']:>3}/{total})   "
              f"{i_pct:>5.1f}% ({stats['Ischemic']:>3}/{total})   "
              f"{n_pct:>5.1f}% ({stats['NoStroke']:>3}/{total})")

print("\n")

# ============================================================================
# STEP 5: ANALYZE CONFIDENCE DISTRIBUTIONS
# ============================================================================

print("Step 5: Analyzing confidence distributions...\n")

for true_class in CLASS_NAMES:
    confidences = predictions_by_class[true_class]['confidences']
    
    if confidences:
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        median_conf = np.median(confidences)
        
        print(f"{true_class}:")
        print(f"  Mean Confidence:   {mean_conf:.4f}")
        print(f"  Median Confidence: {median_conf:.4f}")
        print(f"  Std Deviation:     {std_conf:.4f}")
        print(f"  Min Confidence:    {min_conf:.4f}")
        print(f"  Max Confidence:    {max_conf:.4f}")
        print()

# ============================================================================
# STEP 6: ANALYZE RAW MODEL OUTPUTS (LOGITS)
# ============================================================================

print("Step 6: Analyzing raw model outputs (logits)...\n")

# Group by true class
for true_class in CLASS_NAMES:
    class_outputs = [o for o in all_raw_outputs if o['true_class'] == true_class]
    
    if class_outputs:
        # Average logits for this class
        avg_logits = np.mean([o['logits'] for o in class_outputs], axis=0)
        
        print(f"{true_class} - Average Logits (before softmax):")
        for j, (class_name, logit) in enumerate(zip(CLASS_NAMES, avg_logits)):
            print(f"  {class_name}: {logit:>8.4f}")
        print()

# ============================================================================
# STEP 7: DETECT BIAS PATTERNS
# ============================================================================

print("Step 7: Detecting bias patterns...\n")

bias_indicators = []

# Check 1: Is model always predicting one class?
predictions_list = [o['predicted_class'] for o in all_raw_outputs]
unique_predictions = set(predictions_list)

if len(unique_predictions) == 1:
    bias_indicators.append(f"🚨 CRITICAL: Model predicts ONLY '{unique_predictions.pop()}' for ALL images!")
elif len(unique_predictions) == 2:
    bias_indicators.append(f"⚠️ WARNING: Model predicts only 2 classes: {unique_predictions}")

# Check 2: Is confidence very low for correct class?
for true_class in CLASS_NAMES:
    class_data = [o for o in all_raw_outputs if o['true_class'] == true_class]
    correct_predictions = [o for o in class_data if o['predicted_class'] == true_class]
    
    if correct_predictions:
        correct_confidences = [o['confidence'] for o in correct_predictions]
        avg_correct_conf = np.mean(correct_confidences)
        
        if avg_correct_conf < 0.5:
            bias_indicators.append(
                f"⚠️ WARNING: {true_class} correct predictions have LOW average confidence: {avg_correct_conf:.2%}"
            )

# Check 3: Check class distribution in training vs test
print("📊 Test Data Distribution:")
for class_name in CLASS_NAMES:
    count = test_images[class_name]
    pct = (count / total_images * 100) if total_images > 0 else 0
    print(f"  {class_name}: {count} images ({pct:.1f}%)")
print()

if bias_indicators:
    print("🔴 BIAS INDICATORS DETECTED:\n")
    for indicator in bias_indicators:
        print(f"  {indicator}")
    print()
else:
    print("✅ No major bias indicators detected\n")

# ============================================================================
# STEP 8: CREATE VISUALIZATIONS
# ============================================================================

print("Step 8: Creating visualizations...\n")

# 1. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

confusion = np.zeros((3, 3))
for i, true_class in enumerate(CLASS_NAMES):
    for j, pred_class in enumerate(CLASS_NAMES):
        confusion[i, j] = predictions_by_class[true_class][pred_class]

# Normalize by row (true class)
confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)

sns.heatmap(confusion_normalized, annot=confusion.astype(int), fmt='d',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cmap='Blues', cbar_kws={'label': 'Proportion'},
            ax=ax, annot_kws={'fontsize': 12})

ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix: Model Predictions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bias_confusion_matrix.png'), dpi=300)
print("✅ Saved: bias_confusion_matrix.png")
plt.close()

# 2. Confidence Distribution by Class
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, class_name in enumerate(CLASS_NAMES):
    confidences = predictions_by_class[class_name]['confidences']
    
    if confidences:
        axes[idx].hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{class_name}\n(n={len(confidences)})', fontweight='bold')
        axes[idx].set_xlabel('Confidence')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim([0, 1])
        
        # Add statistics
        mean_conf = np.mean(confidences)
        axes[idx].axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.2%}')
        axes[idx].legend()

plt.suptitle('Confidence Distribution by True Class', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bias_confidence_distribution.png'), dpi=300)
print("✅ Saved: bias_confidence_distribution.png")
plt.close()

# 3. Average Logits by Class
fig, ax = plt.subplots(figsize=(10, 6))

avg_logits_by_class = {}
for true_class in CLASS_NAMES:
    class_outputs = [o for o in all_raw_outputs if o['true_class'] == true_class]
    if class_outputs:
        avg_logits = np.mean([o['logits'] for o in class_outputs], axis=0)
        avg_logits_by_class[true_class] = avg_logits

x = np.arange(len(CLASS_NAMES))
width = 0.25

for i, true_class in enumerate(CLASS_NAMES):
    logits = avg_logits_by_class[true_class]
    offset = (i - 1) * width
    ax.bar(x + offset, logits, width, label=f'True: {true_class}', alpha=0.8)

ax.set_xlabel('Output Class', fontweight='bold')
ax.set_ylabel('Average Logit Value', fontweight='bold')
ax.set_title('Average Raw Model Outputs (Logits) by True Class', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bias_logits_analysis.png'), dpi=300)
print("✅ Saved: bias_logits_analysis.png")
plt.close()

# ============================================================================
# STEP 9: ROOT CAUSE ANALYSIS
# ============================================================================

print("\n")
print("=" * 70)
print("🔍 ROOT CAUSE ANALYSIS")
print("=" * 70)
print()

# Analyze logits to see if one class has consistently higher logits
all_class_avg_logits = {}
for class_idx, class_name in enumerate(CLASS_NAMES):
    logit_values = []
    for output in all_raw_outputs:
        logit_values.append(output['logits'][class_idx])
    all_class_avg_logits[class_name] = np.mean(logit_values)

print("📊 Global Average Logits (ALL images):")
for class_name, logit in sorted(all_class_avg_logits.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {logit:>8.4f}")
print()

# Find the problem
print("🔎 DIAGNOSIS:")
print()

highest_class = max(all_class_avg_logits, key=all_class_avg_logits.get)
highest_logit = all_class_avg_logits[highest_class]

if highest_class == 'NoStroke':
    print(f"⚠️ ISSUE FOUND: '{highest_class}' has the HIGHEST average logit ({highest_logit:.4f})")
    print(f"   This means the model outputs are biased towards 'NoStroke'")
    print()
    print("   Possible causes:")
    print("   1. ❌ Training data was heavily imbalanced (too many NoStroke images)")
    print("   2. ❌ The model was not trained with class weights")
    print("   3. ❌ Loss function didn't properly penalize NoStroke predictions")
    print("   4. ❌ Model trained for too few epochs on stroke classes")
    print("   5. ❌ Augmentation/preprocessing favored NoStroke patterns")
    print()
else:
    print(f"✅ No obvious bias detected. Highest logit is for: {highest_class}")

# ============================================================================
# STEP 10: SAVE DETAILED REPORT
# ============================================================================

print("\nStep 10: Creating detailed report...\n")

# Create CSV with all predictions
df_predictions = pd.DataFrame(all_raw_outputs)
df_predictions_export = df_predictions[['true_class', 'predicted_class', 'confidence', 'image']].copy()
df_predictions_export.to_csv(os.path.join(RESULTS_DIR, 'bias_detailed_predictions.csv'), index=False)
print("✅ Saved: bias_detailed_predictions.csv")

# Summary statistics
summary_stats = {
    'metric': [],
    'value': []
}

for true_class in CLASS_NAMES:
    stats = predictions_by_class[true_class]
    if stats['total'] > 0:
        accuracy = stats[true_class] / stats['total']
        summary_stats['metric'].append(f"{true_class}_accuracy")
        summary_stats['value'].append(f"{accuracy:.2%}")

df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv(os.path.join(RESULTS_DIR, 'bias_summary_stats.csv'), index=False)
print("✅ Saved: bias_summary_stats.csv")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print("\n")
print("=" * 70)
print("💡 RECOMMENDATIONS TO FIX THE BIAS")
print("=" * 70)
print()

recommendations = [
    ("1. Use Class Weights in Training", 
     "If training data was imbalanced, use class_weight='balanced' in PyTorch:"),
    ("   Example: ", 
     "criterion = nn.CrossEntropyLoss(weight=torch.tensor([w1, w2, w3]))"),
    ("", ""),
    ("2. Rebalance Training Data",
     "Oversample stroke images or undersample NoStroke images"),
    ("", ""),
    ("3. Adjust Decision Threshold",
     "Lower the confidence threshold for stroke predictions:"),
    ("   Example:",
     "if prob_stroke > 0.3: predict_stroke else: predict_no_stroke"),
    ("", ""),
    ("4. Data Augmentation",
     "Apply more aggressive augmentation to stroke images during training"),
    ("", ""),
    ("5. Retrain Model",
     "With one of the above approaches and monitor class-wise metrics"),
    ("", ""),
    ("6. Use Focal Loss",
     "Implement focal loss to focus on hard examples"),
]

for metric, recommendation in recommendations:
    if metric:
        print(f"{metric:<25} {recommendation}")
    else:
        print()

print()
print("=" * 70)
print("✅ DIAGNOSIS COMPLETE")
print("=" * 70)
print()
print(f"📁 Results saved to: {RESULTS_DIR}")
print()
