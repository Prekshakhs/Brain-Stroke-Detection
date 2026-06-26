"""Test the balanced validation"""
import cv2
import numpy as np
import os

def is_brain_scan(image_array):
    """Balanced validation"""
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
        
        # Extreme saturation check
        if len(image_array.shape) == 3:
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].astype(float)
            mean_sat = np.mean(saturation)
            std_sat = np.std(saturation)
            max_sat = np.max(saturation)
            
            if mean_sat > 150 and max_sat > 230 and std_sat > 50:
                return False, "Extreme saturation (eye)"
        
        # Skin tone detection for faces
        if len(image_array.shape) == 3:
            r = image_array[:,:,2].astype(float)
            g = image_array[:,:,1].astype(float)
            b = image_array[:,:,0].astype(float)
            
            skin_pixels = 0
            total_pixels = image_array.shape[0] * image_array.shape[1]
            
            for i in range(min(100, image_array.shape[0])):  # Sample check
                for j in range(min(100, image_array.shape[1])):
                    r_val = r[i, j]
                    g_val = g[i, j]
                    b_val = b[i, j]
                    
                    if (50 < r_val < 220 and 40 < g_val < 200 and 20 < b_val < 180 and
                        r_val > g_val > b_val):
                        skin_pixels += 1
            
            skin_ratio = skin_pixels / min(10000, total_pixels)
            
            if skin_ratio > 0.3:
                return False, "Skin tones detected"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)

# Test real brains
print("="*60)
print("REAL BRAIN SCANS")
print("="*60)
pass_count = 0
for class_name in ['NoStroke', 'Hemorrhagic', 'Ischemic']:
    path = f'data/test/{class_name}'
    files = os.listdir(path)[:3]
    for f in files:
        image = cv2.imread(os.path.join(path, f))
        result, msg = is_brain_scan(image)
        status = "✅" if result else "❌"
        if result:
            pass_count += 1
        print(f"{status} {class_name}/{f}: {msg}")

print(f"\nBrain scans: {pass_count}/9 accepted")

# Test synthetic non-medical
print("\n" + "="*60)
print("SYNTHETIC NON-MEDICAL IMAGES")
print("="*60)
reject_count = 0

# Eye image
eye = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(eye, (150, 150), 80, (100, 200, 255), -1)
cv2.circle(eye, (150, 150), 40, (0, 0, 0), -1)
result, msg = is_brain_scan(eye)
status = "❌" if not result else "⚠️"
if not result:
    reject_count += 1
print(f"{status} Eye image: {msg}")

# Colorized pattern (high color variation)
pattern = np.zeros((300, 300, 3), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        b_val = int(100 + 80*np.sin(x/30))
        g_val = int(150 + 50*np.cos(y/30))
        r_val = int(200 + 80*np.sin((x+y)/20))
        pattern[y, x] = [np.clip(b_val, 0, 255), np.clip(g_val, 0, 255), np.clip(r_val, 0, 255)]

result, msg = is_brain_scan(pattern)
status = "❌" if not result else "⚠️"
if not result:
    reject_count += 1
print(f"{status} Colorized pattern: {msg}")

# Natural landscape with color
landscape = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
result, msg = is_brain_scan(landscape)
status = "❌" if not result else "⚠️"
if not result:
    reject_count += 1
print(f"{status} Landscape/noise: {msg}")

print(f"\nNon-medical: {reject_count}/3 rejected")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Medical images accepted: {pass_count}/9 (should be high)")
print(f"Non-medical rejected: {reject_count}/3 (should be high)")
if pass_count >= 8 and reject_count >= 2:
    print("✅ BALANCED - Ready for deployment!")
elif pass_count >= 7:
    print("⚠️ Accepting most medical but missing some non-medical rejection")
else:
    print("❌ Too strict on medical images")
