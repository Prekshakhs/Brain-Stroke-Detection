"""Test grayscale medical brain scan acceptance"""
import cv2
import numpy as np
import os

def is_brain_scan(image_array):
    """Updated validation accepting grayscale medical brain scans"""
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
            
            # Calculate saturation
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].astype(float)
            
            mean_sat = np.mean(saturation)
            std_sat = np.std(saturation)
            max_sat = np.max(saturation)
            
            # Accept grayscale medical images with slight color variations
            # Only reject if artificially colorized (high saturation + concentrated hue)
            if mean_sat > 100:
                hue = hsv[:,:,0]
                hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
                hue_concentration = np.max(hue_hist) / np.sum(hue_hist)
                
                if hue_concentration > 0.25:
                    return False, f"Artificial colorization (sat={mean_sat:.1f})"
            
            if mean_sat > 140 and max_sat > 220 and std_sat > 40:
                return False, "Extreme saturation (eye)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

print("="*80)
print("TESTING GRAYSCALE MEDICAL BRAIN SCAN ACCEPTANCE")
print("="*80)

# Test 1: Real grayscale brain scans
print("\n✓ REAL GRAYSCALE BRAIN SCANS (Should ALL ACCEPT):")
print("-" * 80)
brain_dirs = ["data/test/NoStroke", "data/test/Ischemic", "data/test/Hemorrhagic"]
accept_count = 0
for brain_dir in brain_dirs:
    if os.path.exists(brain_dir):
        files = [f for f in os.listdir(brain_dir) if f.endswith(('.jpg', '.png'))][:2]
        for f in files:
            path = os.path.join(brain_dir, f)
            img = cv2.imread(path)
            if img is not None:
                result, msg = is_brain_scan(img)
                status = "✅" if result else "❌"
                print(f"{status} {brain_dir.split('/')[-1]}: {msg}")
                if result:
                    accept_count += 1

# Test 2: Grayscale medical images with slight color variations
print("\n✓ GRAYSCALE WITH SLIGHT COLOR VARIATIONS (Should ACCEPT):")
print("-" * 80)

grayscale_images = {}

# 1. Pure grayscale (classic CT/MRI)
pure_gray = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        val = int(128 + 60 * np.sin(np.sqrt((i-128)**2 + (j-128)**2) / 30))
        pure_gray[i, j] = [val, val, val]
grayscale_images['Pure Grayscale'] = pure_gray

# 2. Slightly tinted grayscale (slight blue tint)
tinted_blue = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        val = int(128 + 60 * np.sin(np.sqrt((i-128)**2 + (j-128)**2) / 30))
        tinted_blue[i, j] = [val+5, val+5, val+10]  # Slight blue tint
grayscale_images['Slight Blue Tint'] = tinted_blue

# 3. Slightly tinted grayscale (slight sepia/warm tone)
tinted_warm = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        val = int(128 + 60 * np.sin(np.sqrt((i-128)**2 + (j-128)**2) / 30))
        tinted_warm[i, j] = [val+8, val, val-2]  # Warm/sepia tone
grayscale_images['Sepia Tone'] = tinted_warm

gray_accept = 0
for name, img in grayscale_images.items():
    result, msg = is_brain_scan(img)
    status = "✅" if result else "❌"
    print(f"{status} {name:<30}: {msg}")
    if result:
        gray_accept += 1

# Test 3: Artificial colorized images (should reject)
print("\n✗ ARTIFICIAL COLORIZED IMAGES (Should ALL REJECT):")
print("-" * 80)

colorized_images = {}

# 1. Rainbow heatmap
rainbow = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    for j in range(256):
        hue = int(180 * j / 256)
        hsv_pixel = np.array([[[hue, 200, 200]]], dtype=np.uint8)
        rainbow[i, j] = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
colorized_images['Rainbow Heatmap'] = rainbow

# 2. Solid color blocks
blocks = np.zeros((256, 256, 3), dtype=np.uint8)
blocks[:128, :128] = [0, 0, 255]  # Red
blocks[:128, 128:] = [0, 255, 255]  # Yellow
blocks[128:, :128] = [255, 0, 0]  # Blue
blocks[128:, 128:] = [0, 255, 0]  # Green
colorized_images['Colored Blocks'] = blocks

# 3. Red-Blue pattern
redblue = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(0, 256, 32):
    for j in range(0, 256, 32):
        if (i // 32 + j // 32) % 2 == 0:
            redblue[i:i+32, j:j+32] = [0, 0, 255]
        else:
            redblue[i:i+32, j:j+32] = [255, 0, 0]
colorized_images['Red-Blue Pattern'] = redblue

# 4. High saturation yellow
yellow_img = np.ones((256, 256, 3), dtype=np.uint8) * [0, 255, 255]
colorized_images['High Saturation Yellow'] = yellow_img

reject_count = 0
for name, img in colorized_images.items():
    result, msg = is_brain_scan(img)
    status = "❌" if not result else "✅"
    print(f"{status} {name:<30}: {msg}")
    if not result:
        reject_count += 1

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"✅ Real brain scans accepted: {accept_count}/6+")
print(f"✅ Grayscale variations accepted: {gray_accept}/3")
print(f"✅ Artificial colorized rejected: {reject_count}/4")

if accept_count >= 2 and gray_accept == 3 and reject_count == 4:
    print("\n🎉 GRAYSCALE ACCEPTANCE WORKING CORRECTLY!")
else:
    print(f"\n⚠️  Some adjustments may be needed.")
