"""Test the v6 brain scan validation logic"""
import cv2
import numpy as np
import os
from pathlib import Path

def is_brain_scan(image):
    """
    Comprehensive brain scan validation using multi-criterion rejection approach.
    Returns: (is_valid, message)
    """
    if image is None or image.size == 0:
        return False, "Invalid image"
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    
    # 1. DIMENSION CHECKS
    if height < 80 or width < 80 or height > 4096 or width > 4096:
        return False, "Image dimensions outside acceptable range (80-4096 pixels)"
    
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 3.0:
        return False, "Image aspect ratio too extreme (>3.0)"
    
    # 2. EDGE DETECTION (Canny)
    edges = cv2.Canny(gray, 30, 100)
    edge_ratio = np.sum(edges > 0) / (height * width)
    if edge_ratio < 0.0005 or edge_ratio > 0.3:
        return False, f"Edge ratio {edge_ratio:.4f} outside medical range (0.05%-30%)"
    
    # 3. HISTOGRAM ENTROPY (grayscale natural image detection)
    hist, _ = np.histogram(gray, 256, [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    hist_entropy = -np.sum(hist * np.log2(hist))
    
    if hist_entropy > 5.0:
        return False, f"High histogram entropy {hist_entropy:.2f} suggests natural image"
    
    # 4. SATURATION & COLOR ANALYSIS
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(float)
        
        mean_sat = np.mean(saturation)
        max_sat = np.max(saturation)
        std_sat = np.std(saturation)
        
        # EXTREME saturation detection (eyes only)
        if mean_sat > 150 and max_sat > 230 and std_sat > 50:
            return False, "Image has extreme eye-like color characteristics"
        
        # Channel ratio analysis (faces with extreme color)
        b, g, r = cv2.split(image)
        channel_diff = np.max([
            np.abs(r.astype(float) - g.astype(float)),
            np.abs(g.astype(float) - b.astype(float)),
            np.abs(r.astype(float) - b.astype(float))
        ])
        channel_ratio = np.mean(channel_diff)
        
        if channel_ratio > 15:
            return False, "Image has extreme color variation (likely face/artificial)"
    
    # LAPLACIAN VARIANCE (fine texture detection - faces/eyes)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    if laplacian_var > 4500:
        return False, f"High texture variance {laplacian_var:.0f} suggests face/natural image"
    
    # 6. UNIQUE GRAY LEVELS
    unique_values = len(np.unique(gray))
    if unique_values < 15:
        return False, f"Too few unique gray levels ({unique_values}), likely simple pattern"
    
    # 7. CONTOUR ANALYSIS
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        contour_ratio = total_contour_area / (height * width)
        
        if contour_ratio < 0.02 or contour_ratio > 0.99:
            return False, f"Contour ratio {contour_ratio:.4f} outside medical range"
        
        # Circularity check (perfect circles are eyes)
        for contour in contours:
            if len(contour) >= 5:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                    if circularity > 0.94:
                        return False, f"Perfect circular contour (circularity {circularity:.2f}) detected - likely eye"
    
    # 8. CENTER REGION VARIANCE (medical tissue validation)
    h_start, h_end = height // 4, (3 * height) // 4
    w_start, w_end = width // 4, (3 * width) // 4
    center_region = gray[h_start:h_end, w_start:w_end]
    center_var = np.var(center_region)
    
    if center_var < 100 or center_var > 70000:
        return False, f"Center region variance {center_var:.0f} outside medical range (100-70000)"
    
    # ALL CHECKS PASSED
    return True, "Valid brain scan"


# Test with actual brain images
test_classes = ['NoStroke', 'Hemorrhagic', 'Ischemic']
results = {}

for class_name in test_classes:
    test_dir = f'data/test/{class_name}'
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found")
        continue
    
    images = os.listdir(test_dir)[:3]  # Test first 3 of each class
    results[class_name] = {'pass': 0, 'fail': 0}
    
    for img_file in images:
        img_path = os.path.join(test_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is not None:
            is_valid, msg = is_brain_scan(image)
            if is_valid:
                results[class_name]['pass'] += 1
            else:
                results[class_name]['fail'] += 1
                print(f"FAILED {class_name}/{img_file}: {msg}")

print("\n" + "="*60)
print("VALIDATION TEST RESULTS (v6 Multi-Criterion)")
print("="*60)
for class_name, result in results.items():
    total = result['pass'] + result['fail']
    rate = (result['pass'] / total * 100) if total > 0 else 0
    print(f"{class_name:15} - PASS: {result['pass']}/{total} ({rate:.0f}%)")

print("\n✅ If all brain classes show 100% acceptance, v6 validation is working correctly!")
