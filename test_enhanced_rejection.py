"""Test enhanced rejection of non-medical images"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

def analyze_image(img_array):
    """Analyze image characteristics"""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    # Laplacian (texture complexity)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    # Entropy (information content)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    
    # Color characteristics
    if len(img_array.shape) == 3:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1].astype(float)
        mean_sat = np.mean(saturation)
        max_sat = np.max(saturation)
        std_sat = np.std(saturation)
        
        # Hue distribution
        hue = hsv[:,:,0]
        hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        dominant_hue = np.argmax(hue_hist)
    else:
        mean_sat = max_sat = std_sat = 0
        dominant_hue = 0
    
    # Unique values
    unique_vals = len(np.unique(gray))
    
    # Directional edges (horizontal vs vertical)
    edges_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_ratio = np.sum(np.abs(edges_h) > 50) / (gray.shape[0] * gray.shape[1])
    v_ratio = np.sum(np.abs(edges_v) > 50) / (gray.shape[0] * gray.shape[1])
    
    return {
        'edge_ratio': edge_ratio,
        'laplacian_var': laplacian_var,
        'entropy': entropy,
        'mean_sat': mean_sat,
        'max_sat': max_sat,
        'std_sat': std_sat,
        'dominant_hue': dominant_hue,
        'unique_vals': unique_vals,
        'h_ratio': h_ratio,
        'v_ratio': v_ratio
    }

# Create test images
print("="*70)
print("ANALYZING NON-MEDICAL IMAGE CHARACTERISTICS")
print("="*70)

test_cases = {}

# 1. Natural photograph (landscape)
landscape = np.zeros((512, 512, 3), dtype=np.uint8)
# Sky
landscape[0:250, :] = [135, 206, 235]  # Sky blue
# Grass
landscape[250:512, :] = [34, 139, 34]  # Green
# Add some variation
for i in range(100):
    x, y = np.random.randint(0, 512), np.random.randint(0, 512)
    landscape[max(0, y-5):min(512, y+5), max(0, x-5):min(512, x+5)] = [
        np.random.randint(100, 200),
        np.random.randint(100, 200),
        np.random.randint(100, 200)
    ]
test_cases['Landscape/Nature'] = landscape

# 2. Face/Portrait photo (synthetic)
face = np.zeros((512, 512, 3), dtype=np.uint8)
# Skin tone background
face[:, :] = [180, 120, 100]  # Skin tone
# Add face features
cv2.circle(face, (256, 200), 100, (160, 100, 80), -1)  # Face circle
cv2.circle(face, (230, 180), 20, (50, 50, 50), -1)  # Eye
cv2.circle(face, (282, 180), 20, (50, 50, 50), -1)  # Eye
cv2.line(face, (256, 220), (256, 240), (200, 100, 100), 5)  # Nose
test_cases['Face/Portrait'] = face

# 3. Eye close-up (synthetic)
eye_closeup = np.zeros((512, 512, 3), dtype=np.uint8)
eye_closeup[:, :] = [180, 120, 100]  # Skin tone background
cv2.circle(eye_closeup, (256, 256), 150, (50, 100, 150), -1)  # Eye white/color
cv2.circle(eye_closeup, (256, 256), 80, (20, 20, 20), -1)  # Pupil
cv2.circle(eye_closeup, (270, 240), 30, (255, 200, 200), -1)  # Highlight (extreme color)
test_cases['Eye Close-up'] = eye_closeup

# 4. Text/Document
document = np.ones((512, 512, 3), dtype=np.uint8) * 255
cv2.putText(document, "NOT A BRAIN SCAN", (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
cv2.rectangle(document, (20, 20), (492, 492), (0, 0, 0), 3)
test_cases['Document/Text'] = document

# 5. Random noise
noise = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
test_cases['Random Noise'] = noise

# 6. Geometric pattern (red/blue alternating)
pattern = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(0, 512, 32):
    for j in range(0, 512, 32):
        if (i // 32 + j // 32) % 2 == 0:
            pattern[i:i+32, j:j+32] = [255, 0, 0]  # Blue
        else:
            pattern[i:i+32, j:j+32] = [0, 0, 255]  # Red
test_cases['Geometric Pattern'] = pattern

# 7. X-ray of chest (non-brain medical)
chest_xray = np.random.randint(50, 150, (512, 512), dtype=np.uint8)
chest_xray = cv2.cvtColor(chest_xray, cv2.COLOR_GRAY2BGR)
# Add some features
cv2.ellipse(chest_xray, (256, 250), (150, 200), 0, 0, 360, (50, 50, 50), 3)  # Chest outline
cv2.ellipse(chest_xray, (220, 250), (60, 80), 0, 0, 360, (30, 30, 30), 2)  # Left lung
cv2.ellipse(chest_xray, (292, 250), (60, 80), 0, 0, 360, (30, 30, 30), 2)  # Right lung
test_cases['Chest X-ray'] = chest_xray

# Analyze each
print("\n{:<25} {:<12} {:<12} {:<10} {:<10} {:<10} {:<15}".format(
    "Image Type", "Edge%", "Laplacian", "Entropy", "Mean_Sat", "Edge_Ratio", "Unique_Vals"
))
print("-" * 100)

for name, img in test_cases.items():
    stats = analyze_image(img)
    print("{:<25} {:<12.4f} {:<12.1f} {:<10.2f} {:<10.1f} {:<10.4f} {:<15}".format(
        name,
        stats['h_ratio'] * 100,
        stats['laplacian_var'],
        stats['entropy'],
        stats['mean_sat'],
        stats['edge_ratio'],
        stats['unique_vals']
    ))

print("\n" + "="*70)
print("REAL BRAIN SCAN REFERENCE (for comparison)")
print("="*70)

# Load a real brain scan for comparison
import os
brain_dir = "data/test/NoStroke"
if os.path.exists(brain_dir):
    brain_files = [f for f in os.listdir(brain_dir) if f.endswith(('.jpg', '.png'))]
    if brain_files:
        brain_path = os.path.join(brain_dir, brain_files[0])
        real_brain = cv2.imread(brain_path)
        if real_brain is not None:
            stats = analyze_image(real_brain)
            print("{:<25} {:<12.4f} {:<12.1f} {:<10.2f} {:<10.1f} {:<10.4f} {:<15}".format(
                "Real Brain Scan",
                stats['h_ratio'] * 100,
                stats['laplacian_var'],
                stats['entropy'],
                stats['mean_sat'],
                stats['edge_ratio'],
                stats['unique_vals']
            ))

print("\n" + "="*70)
print("REJECTION STRATEGY")
print("="*70)
print("""
CURRENT CHECKS:
1. Skin tone detection (R>G>B, 50-220 range, >30%) → Rejects faces/animals
2. Extreme saturation (mean>150, max>230, std>50) → Rejects eyes
3. Blank/no structure checks → Rejects empty images

RECOMMENDED ADDITIONAL CHECKS:
1. Laplacian variance: Brain scans have moderate complexity (500-4000)
   - Too low (<200) = texture-less (document, solid colors)
   - Too high (>6000) = highly textured (natural photos, noise)

2. Horizontal line ratio: Natural photos have more varied edge directions
   - Brain scans: more balanced h_ratio/v_ratio (circular/oval brain)
   - Natural: often have clear horizontal lines (horizon)

3. Hue distribution: Brain scans (grayscale/medical) have low hue variance
   - Non-medical color photos: high hue variance

4. Edge ratio distribution: Brain scans have consistent edges throughout
   - Natural photos: edges concentrated at boundaries (sky-grass, etc.)
""")
