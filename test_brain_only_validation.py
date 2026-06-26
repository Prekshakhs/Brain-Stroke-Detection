"""Test brain-only validation"""
import cv2
import numpy as np
import os

def is_brain_scan(image_array):
    """Brain scan validation - accepts only medical brain images"""
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
            return False, f"Excessive edges: {edge_ratio:.2%}"
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        if laplacian_var < 250:
            return False, f"Low texture: {laplacian_var:.0f}"
        
        if laplacian_var > 5000:
            return False, f"High texture: {laplacian_var:.0f}"
        
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
                return False, f"Skin tones: {skin_ratio:.1%}"
            
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].astype(float)
            
            mean_sat = np.mean(saturation)
            std_sat = np.std(saturation)
            max_sat = np.max(saturation)
            
            if mean_sat > 140 and max_sat > 220 and std_sat > 40:
                return False, "Extreme saturation"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

print("="*80)
print("TESTING BRAIN-ONLY VALIDATION")
print("="*80)

# Test 1: Real brain scans
print("\n✅ REAL BRAIN SCANS (Should ACCEPT):")
print("-" * 80)
brain_dirs = ["data/test/NoStroke", "data/test/Ischemic", "data/test/Hemorrhagic"]
real_count = 0
for brain_dir in brain_dirs:
    if os.path.exists(brain_dir):
        files = [f for f in os.listdir(brain_dir) if f.endswith(('.jpg', '.png'))][:2]
        for f in files:
            path = os.path.join(brain_dir, f)
            img = cv2.imread(path)
            if img is not None:
                result, msg = is_brain_scan(img)
                status = "✅" if result else "❌"
                print(f"  {status} {brain_dir.split('/')[-1]}: {msg}")
                if result:
                    real_count += 1

# Test 2: Non-medical synthetic images
print("\n❌ NON-MEDICAL IMAGES (Should REJECT):")
print("-" * 80)

test_images = {}

# Landscape
landscape = np.zeros((512, 512, 3), dtype=np.uint8)
landscape[0:250, :] = [135, 206, 235]
landscape[250:512, :] = [34, 139, 34]
test_images['Landscape'] = landscape

# Face
face = np.zeros((512, 512, 3), dtype=np.uint8)
face[:, :] = [180, 120, 100]
cv2.circle(face, (256, 200), 100, (160, 100, 80), -1)
cv2.circle(face, (230, 180), 20, (50, 50, 50), -1)
cv2.circle(face, (282, 180), 20, (50, 50, 50), -1)
test_images['Face'] = face

# Eye
eye = np.zeros((512, 512, 3), dtype=np.uint8)
eye[:, :] = [180, 120, 100]
cv2.circle(eye, (256, 256), 150, (50, 100, 150), -1)
cv2.circle(eye, (256, 256), 80, (20, 20, 20), -1)
cv2.circle(eye, (270, 240), 30, (255, 200, 200), -1)
test_images['Eye'] = eye

# Document
document = np.ones((512, 512, 3), dtype=np.uint8) * 255
cv2.putText(document, "NOT BRAIN", (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
test_images['Document'] = document

# Noise
noise = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
test_images['Noise'] = noise

# Pattern
pattern = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(0, 512, 32):
    for j in range(0, 512, 32):
        if (i // 32 + j // 32) % 2 == 0:
            pattern[i:i+32, j:j+32] = [255, 0, 0]
        else:
            pattern[i:i+32, j:j+32] = [0, 0, 255]
test_images['Colored Pattern'] = pattern

# Solid color
solid = np.ones((512, 512, 3), dtype=np.uint8) * 128
test_images['Solid Color'] = solid

reject_count = 0
for name, img in test_images.items():
    result, msg = is_brain_scan(img)
    status = "❌" if not result else "✅"
    print(f"  {status} {name:<20}: {msg}")
    if not result:
        reject_count += 1

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"✅ Real brain scans accepted: {real_count}/4")
print(f"❌ Non-medical images rejected: {reject_count}/7")

if real_count >= 3 and reject_count == 7:
    print("\n🎉 BRAIN-ONLY VALIDATION SUCCESSFUL!")
else:
    print(f"\n⚠️  Check validation logic")
