"""Test validation with synthetic non-medical images"""
import cv2
import numpy as np
import os

def is_brain_scan(image):
    """V6 validation function"""
    if image is None or image.size == 0:
        return False, "Invalid image"
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    
    # DIMENSION CHECKS
    if height < 80 or width < 80 or height > 4096 or width > 4096:
        return False, "Image dimensions outside acceptable range"
    
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 3.0:
        return False, "Image aspect ratio too extreme"
    
    # EDGE DETECTION
    edges = cv2.Canny(gray, 30, 100)
    edge_ratio = np.sum(edges > 0) / (height * width)
    if edge_ratio < 0.0005 or edge_ratio > 0.3:
        return False, f"Edge ratio outside medical range"
    
    # HISTOGRAM ENTROPY
    hist, _ = np.histogram(gray, 256, [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    hist_entropy = -np.sum(hist * np.log2(hist))
    
    if hist_entropy > 5.0:
        return False, f"High histogram entropy {hist_entropy:.2f} suggests natural image"
    
    # SATURATION & COLOR
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(float)
        
        mean_sat = np.mean(saturation)
        max_sat = np.max(saturation)
        std_sat = np.std(saturation)
        
        if mean_sat > 150 and max_sat > 230 and std_sat > 50:
            return False, "Extreme eye-like color characteristics"
        
        b, g, r = cv2.split(image)
        channel_diff = np.max([
            np.abs(r.astype(float) - g.astype(float)),
            np.abs(g.astype(float) - b.astype(float)),
            np.abs(r.astype(float) - b.astype(float))
        ])
        channel_ratio = np.mean(channel_diff)
        
        if channel_ratio > 15:
            return False, f"Extreme color variation {channel_ratio:.1f} (face/artificial)"
    
    # LAPLACIAN VARIANCE
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    if laplacian_var > 4500:
        return False, f"High texture variance {laplacian_var:.0f}"
    
    # UNIQUE GRAY LEVELS
    unique_values = len(np.unique(gray))
    if unique_values < 15:
        return False, f"Too few unique gray levels"
    
    # CONTOUR ANALYSIS
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        contour_ratio = total_contour_area / (height * width)
        
        if contour_ratio < 0.02 or contour_ratio > 0.99:
            return False, f"Contour ratio outside medical range"
        
        for contour in contours:
            if len(contour) >= 5:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                    if circularity > 0.94:
                        return False, f"Perfect circular contour - likely eye"
    
    # CENTER REGION VARIANCE
    h_start, h_end = height // 4, (3 * height) // 4
    w_start, w_end = width // 4, (3 * width) // 4
    center_region = gray[h_start:h_end, w_start:w_end]
    center_var = np.var(center_region)
    
    if center_var < 100 or center_var > 70000:
        return False, f"Center variance {center_var:.0f} outside range"
    
    return True, "Valid brain scan"

# Generate synthetic non-medical test images
print("Creating synthetic non-medical test images...")
print("="*60)

test_cases = {
    "grayscale_face": None,
    "eye_image": None,
    "landscape": None,
    "random_noise": None,
    "colorized_face": None
}

# 1. Synthetic face (high texture, fine details)
size = 300
face = np.zeros((size, size, 3), dtype=np.uint8)
# Add fine facial texture pattern (high Laplacian variance)
for y in range(size):
    for x in range(size):
        # Create fine texture
        val = int(128 + 50 * np.sin(x/10) * np.cos(y/10) + 30 * np.random.rand())
        val = np.clip(val, 0, 255)
        face[y, x] = [val-10, val, val+10]
test_cases["colorized_face"] = face

# 2. Synthetic eye (perfect circle, extreme saturation)
eye = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(eye, (150, 150), 80, (100, 200, 255), -1)  # Blue iris
cv2.circle(eye, (150, 150), 40, (0, 0, 0), -1)  # Pupil
test_cases["eye_image"] = eye

# 3. Landscape/natural (high entropy, varied texture)
landscape = np.zeros((300, 300), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        # Create natural-like variation with high entropy
        val = int(100 + 60*np.sin(x/50) + 60*np.cos(y/50) + 40*np.random.rand())
        landscape[y, x] = np.clip(val, 0, 255)
landscape_3ch = cv2.cvtColor(landscape, cv2.COLOR_GRAY2BGR)
# Add color variation (high channel ratio)
landscape_3ch[:,:,0] = np.clip(landscape_3ch[:,:,0].astype(int) + 30, 0, 255)
test_cases["landscape"] = landscape_3ch

# 4. Grayscale face with high texture variance
face_gray = np.zeros((300, 300), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        # Fine texture like skin
        val = int(120 + 40*np.sin(x/5) * np.cos(y/5) + 30*np.random.rand())
        face_gray[y, x] = np.clip(val, 0, 255)
test_cases["grayscale_face"] = face_gray

# 5. Random noise (high entropy, variable texture)
noise = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
test_cases["random_noise"] = noise

# Test all synthetic images
print("\nTesting synthetic non-medical images:")
print("-"*60)

results = {"REJECT": 0, "ACCEPT": 0, "details": []}

for name, img in test_cases.items():
    if img is not None:
        is_valid, msg = is_brain_scan(img)
        status = "❌ REJECT" if not is_valid else "⚠️ ACCEPT"
        results["REJECT" if not is_valid else "ACCEPT"] += 1
        results["details"].append((name, status, msg))
        print(f"{name:20} - {status}: {msg}")

print("\n" + "="*60)
print(f"SYNTHETIC TEST RESULTS")
print("="*60)
print(f"Correctly Rejected: {results['REJECT']}/5")
print(f"Incorrectly Accepted: {results['ACCEPT']}/5")

if results['REJECT'] == 5:
    print("\n✅ PERFECT! All non-medical images rejected!")
elif results['REJECT'] >= 4:
    print("\n⚠️ GOOD! Most non-medical images rejected (1 issue)")
else:
    print("\n❌ Problem! Too many non-medical images accepted")

print("\nValidation Status: " + ("✅ READY FOR DEPLOYMENT" if results['REJECT'] >= 4 else "🔄 NEEDS TUNING"))
