"""Debug why grayscale brains are being rejected"""
import cv2
import numpy as np

# Load a real brain scan
image = cv2.imread('data/test/NoStroke/10001_12.jpg')
if image is None:
    print("ERROR: Could not load image")
    exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

print("="*60)
print("DEBUGGING GRAYSCALE BRAIN REJECTION")
print("="*60)

# 1. Dimensions
print(f"\n1. DIMENSIONS")
print(f"   Size: {w}x{h}")
print(f"   Valid: {w >= 80 and h >= 80 and w <= 4096 and h <= 4096}")

# 2. Aspect ratio
aspect = max(w, h) / min(w, h)
print(f"\n2. ASPECT RATIO")
print(f"   Ratio: {aspect:.2f}")
print(f"   Valid (< 3.0): {aspect < 3.0}")

# 3. Edge detection
edges = cv2.Canny(gray, 30, 100)
edge_ratio = np.sum(edges > 0) / (h * w)
print(f"\n3. EDGE DETECTION")
print(f"   Edge ratio: {edge_ratio:.4f} ({edge_ratio*100:.2f}%)")
print(f"   Valid (0.05%-30%): {0.0005 < edge_ratio < 0.3}")
if not (0.0005 < edge_ratio < 0.3):
    print(f"   REJECTED HERE: edge_ratio {edge_ratio:.4f} not in (0.0005, 0.3)")

# 4. Histogram entropy
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_norm = hist.flatten() / (hist.sum() + 1e-10)
hist_clean = hist_norm[hist_norm > 0]
entropy = -np.sum(hist_clean * np.log2(hist_clean))
print(f"\n4. HISTOGRAM ENTROPY")
print(f"   Entropy: {entropy:.2f}")
print(f"   Valid (1.5-5.0): {1.5 < entropy < 5.0}")
if not (1.5 < entropy < 5.0):
    print(f"   REJECTED HERE: entropy {entropy:.2f} not in (1.5, 5.0)")

# 5. Unique gray levels
unique = len(np.unique(gray))
print(f"\n5. UNIQUE GRAY LEVELS")
print(f"   Count: {unique}")
print(f"   Valid (>= 15): {unique >= 15}")

# 6. Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
lap_var = np.var(laplacian)
print(f"\n6. LAPLACIAN VARIANCE")
print(f"   Variance: {lap_var:.0f}")
print(f"   Valid (<= 4500): {lap_var <= 4500}")

# 7. Center variance
cy, cx = h // 2, w // 2
quarter = min(h, w) // 4
center_region = gray[cy-quarter:cy+quarter, cx-quarter:cx+quarter]
center_var = np.var(center_region)
print(f"\n7. CENTER REGION VARIANCE")
print(f"   Variance: {center_var:.0f}")
print(f"   Valid (100-70000): {100 < center_var < 70000}")

print("\n" + "="*60)
print("OVERALL RESULT")
print("="*60)
all_pass = (
    w >= 80 and h >= 80 and w <= 4096 and h <= 4096 and
    aspect < 3.0 and
    0.0005 < edge_ratio < 0.3 and
    1.5 < entropy < 5.0 and
    unique >= 15 and
    lap_var <= 4500 and
    100 < center_var < 70000
)
print(f"Should ACCEPT: {all_pass}")
if not all_pass:
    print("\nFirst failure:")
    if not (w >= 80 and h >= 80 and w <= 4096 and h <= 4096):
        print("  - Dimensions check")
    elif not aspect < 3.0:
        print("  - Aspect ratio check")
    elif not (0.0005 < edge_ratio < 0.3):
        print(f"  - Edge ratio check: {edge_ratio:.4f}")
    elif not (1.5 < entropy < 5.0):
        print(f"  - Entropy check: {entropy:.2f}")
    elif not unique >= 15:
        print(f"  - Unique levels check: {unique}")
    elif not lap_var <= 4500:
        print(f"  - Laplacian check: {lap_var:.0f}")
    elif not (100 < center_var < 70000):
        print(f"  - Center variance check: {center_var:.0f}")
