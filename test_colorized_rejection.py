"""Test colorized image rejection"""
import cv2
import numpy as np
import os

def is_brain_scan(image_array):
    """Updated brain scan validation with colorized rejection"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        height, width = gray.shape
        
        if width < 80 or height < 80:
            return False, "Too small"
        if width > 4096 or height > 4096:
            return False, "Too large"
        
        unique_vals = len(np.unique(gray))
        if unique_vals < 5:
            return False, "Blank"
        
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (height * width)
        
        if edge_ratio < 0.001:
            return False, "No structure"
        
        if edge_ratio > 0.15:
            return False, "Excessive edges"
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        if laplacian_var < 250:
            return False, "Low texture"
        
        if laplacian_var > 5000:
            return False, "High texture"
        
        if len(image_array.shape) == 3:
            r = image_array[:,:,2].astype(float)
            g = image_array[:,:,1].astype(float)
            b = image_array[:,:,0].astype(float)
            
            skin_pixels = 0
            total_pixels = image_array.shape[0] * image_array.shape[1]
            
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    r_val = r[i, j]
                    g_val = g[i, j]
                    b_val = b[i, j]
                    
                    if (50 < r_val < 220 and 40 < g_val < 200 and 20 < b_val < 180 and
                        r_val > g_val > b_val):
                        skin_pixels += 1
            
            skin_ratio = skin_pixels / total_pixels
            if skin_ratio > 0.25:
                return False, "Faces/animals"
            
            # Calculate saturation to detect colorization
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].astype(float)
            
            mean_sat = np.mean(saturation)
            std_sat = np.std(saturation)
            max_sat = np.max(saturation)
            
            # REJECT: Colorized images (mean_sat > 50)
            if mean_sat > 50:
                return False, f"Colorized: mean_sat={mean_sat:.1f}"
            
            if mean_sat > 140 and max_sat > 220 and std_sat > 40:
                return False, "Extreme saturation (eye)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

print("="*80)
print("TESTING COLORIZED IMAGE REJECTION")
print("="*80)

# Test 1: Real brain scans (should accept)
print("\n✓ REAL BRAIN SCANS (Grayscale - Should ACCEPT):")
print("-" * 80)
brain_dirs = ["data/test/NoStroke", "data/test/Ischemic", "data/test/Hemorrhagic"]
accept_count = 0
for brain_dir in brain_dirs:
    if os.path.exists(brain_dir):
        files = [f for f in os.listdir(brain_dir) if f.endswith(('.jpg', '.png'))][:1]
        for f in files:
            path = os.path.join(brain_dir, f)
            img = cv2.imread(path)
            if img is not None:
                result, msg = is_brain_scan(img)
                status = "✅" if result else "❌"
                print(f"{status} {brain_dir.split('/')[-1]}: {msg}")
                if result:
                    accept_count += 1

# Test 2: Colorized images (should reject)
print("\n✗ COLORIZED/ARTIFICIAL IMAGES (Should ALL REJECT):")
print("-" * 80)

test_images = {}

# 1. Blue-red colorized heatmap
heatmap = np.zeros((256, 256, 3), dtype=np.uint8)
heatmap[:128, :] = [255, 0, 0]  # Red
heatmap[128:, :] = [0, 0, 255]  # Blue
test_images['Red-Blue Heatmap'] = heatmap

# 2. Rainbow gradient
rainbow = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    hue = int(180 * i / 256)
    rainbow[:, i] = [hue, 255, 255]  # HSV -> RGB
    rainbow[:, i] = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
test_images['Rainbow Gradient'] = rainbow

# 3. Colorful checkerboard
checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(0, 256, 32):
    for j in range(0, 256, 32):
        if (i // 32 + j // 32) % 2 == 0:
            checkerboard[i:i+32, j:j+32] = [0, 0, 255]  # Red
        else:
            checkerboard[i:i+32, j:j+32] = [255, 0, 0]  # Blue
test_images['Colorful Checkerboard'] = checkerboard

# 4. Yellow-green gradient
grad = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    grad[i, :] = [0, int(255 * (1 - i/256)), int(255 * i/256)]  # Yellow to green
test_images['Yellow-Green Gradient'] = grad

# 5. Colorized face
colorized_face = np.zeros((256, 256, 3), dtype=np.uint8)
colorized_face[:, :] = [200, 100, 80]  # Skin tone but saturated
cv2.circle(colorized_face, (128, 100), 50, (100, 50, 30), -1)
cv2.circle(colorized_face, (110, 90), 10, (0, 0, 255), -1)
cv2.circle(colorized_face, (146, 90), 10, (0, 0, 255), -1)
test_images['Colorized Face'] = colorized_face

# 6. Landscape with high colors
landscape = np.zeros((256, 256, 3), dtype=np.uint8)
landscape[:128, :] = [120, 200, 255]  # Sky blue
landscape[128:, :] = [0, 150, 0]  # Grass green
test_images['Colored Landscape'] = landscape

reject_count = 0
for name, img in test_images.items():
    result, msg = is_brain_scan(img)
    status = "❌" if not result else "✅"
    print(f"{status} {name:<30}: {msg}")
    if not result:
        reject_count += 1

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"✅ Real brain scans accepted: {accept_count}/3+")
print(f"✅ Colorized images rejected: {reject_count}/6")
if accept_count >= 1 and reject_count == 6:
    print("\n🎉 COLORIZED REJECTION WORKING CORRECTLY!")
else:
    print(f"\n⚠️  Some tests need adjustment.")
