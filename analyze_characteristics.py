"""Analyze characteristics of real brain scans vs synthetic faces"""
import cv2
import numpy as np
import os

def analyze_image(img_path, label):
    """Extract detailed characteristics"""
    image = cv2.imread(img_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    # Entropy
    hist, _ = np.histogram(gray, 256, [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Center variance
    h, w = gray.shape
    h_start, h_end = h // 4, (3 * h) // 4
    w_start, w_end = w // 4, (3 * w) // 4
    center_region = gray[h_start:h_end, w_start:w_end]
    center_var = np.var(center_region)
    
    # Edges
    edges = cv2.Canny(gray, 30, 100)
    edge_ratio = np.sum(edges > 0) / (h * w)
    
    # Std dev
    std_dev = np.std(gray)
    
    # Contrast (difference between max and min)
    contrast = np.max(gray) - np.min(gray)
    
    # Local patterns - standard deviation of local regions
    local_stds = []
    for i in range(0, h-32, 16):
        for j in range(0, w-32, 16):
            region = gray[i:i+32, j:j+32]
            local_stds.append(np.std(region))
    avg_local_std = np.mean(local_stds) if local_stds else 0
    
    return {
        'label': label,
        'path': img_path,
        'laplacian_var': laplacian_var,
        'entropy': entropy,
        'center_var': center_var,
        'edge_ratio': edge_ratio,
        'std_dev': std_dev,
        'contrast': contrast,
        'avg_local_std': avg_local_std
    }

# Analyze real brain scans
print("="*80)
print("ANALYZING REAL BRAIN SCANS")
print("="*80)

brain_stats = []
for class_name in ['NoStroke', 'Hemorrhagic', 'Ischemic']:
    test_dir = f'data/test/{class_name}'
    if not os.path.exists(test_dir):
        continue
    
    images = os.listdir(test_dir)[:5]  # First 5
    print(f"\n{class_name.upper()}:")
    print("-" * 80)
    
    for img_file in images:
        img_path = os.path.join(test_dir, img_file)
        stats = analyze_image(img_path, class_name)
        if stats:
            brain_stats.append(stats)
            print(f"  {img_file:30} | Laplacian: {stats['laplacian_var']:8.0f} | "
                  f"Entropy: {stats['entropy']:5.2f} | Center: {stats['center_var']:8.0f} | "
                  f"Edges: {stats['edge_ratio']:6.4f} | LocalStd: {stats['avg_local_std']:6.1f}")

# Calculate averages
print("\n" + "="*80)
print("BRAIN SCAN STATISTICS (AVERAGE)")
print("="*80)
if brain_stats:
    avg_laplacian = np.mean([s['laplacian_var'] for s in brain_stats])
    avg_entropy = np.mean([s['entropy'] for s in brain_stats])
    avg_center = np.mean([s['center_var'] for s in brain_stats])
    avg_edges = np.mean([s['edge_ratio'] for s in brain_stats])
    avg_local_std = np.mean([s['avg_local_std'] for s in brain_stats])
    
    print(f"Laplacian Variance:   {avg_laplacian:8.0f} (range: {min([s['laplacian_var'] for s in brain_stats]):8.0f} - {max([s['laplacian_var'] for s in brain_stats]):8.0f})")
    print(f"Entropy:              {avg_entropy:8.2f} (range: {min([s['entropy'] for s in brain_stats]):8.2f} - {max([s['entropy'] for s in brain_stats]):8.2f})")
    print(f"Center Variance:      {avg_center:8.0f} (range: {min([s['center_var'] for s in brain_stats]):8.0f} - {max([s['center_var'] for s in brain_stats]):8.0f})")
    print(f"Edge Ratio:           {avg_edges:8.4f} (range: {min([s['edge_ratio'] for s in brain_stats]):8.4f} - {max([s['edge_ratio'] for s in brain_stats]):8.4f})")
    print(f"Avg Local Std Dev:    {avg_local_std:8.1f} (range: {min([s['avg_local_std'] for s in brain_stats]):8.1f} - {max([s['avg_local_std'] for s in brain_stats]):8.1f})")

# Now analyze synthetic face
print("\n" + "="*80)
print("SYNTHETIC GRAYSCALE FACE ANALYSIS")
print("="*80)

face_gray = np.zeros((300, 300), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        val = int(120 + 40*np.sin(x/5) * np.cos(y/5) + 30*np.random.rand())
        face_gray[y, x] = np.clip(val, 0, 255)

# Save it temporarily
cv2.imwrite('temp_face.jpg', face_gray)
face_stats = analyze_image('temp_face.jpg', 'SyntheticFace')

print(f"Synthetic Face Characteristics:")
print(f"  Laplacian Variance:   {face_stats['laplacian_var']:8.0f}")
print(f"  Entropy:              {face_stats['entropy']:8.2f}")
print(f"  Center Variance:      {face_stats['center_var']:8.0f}")
print(f"  Edge Ratio:           {face_stats['edge_ratio']:8.4f}")
print(f"  Avg Local Std Dev:    {face_stats['avg_local_std']:8.1f}")

# Compare and recommend thresholds
print("\n" + "="*80)
print("THRESHOLD RECOMMENDATIONS")
print("="*80)
print(f"\nLaplacian Variance:")
print(f"  Brain avg:  {avg_laplacian:.0f}")
print(f"  Face:       {face_stats['laplacian_var']:.0f}")
print(f"  Recommendation: Use threshold ~{avg_laplacian + (face_stats['laplacian_var'] - avg_laplacian) * 0.5:.0f}")
print(f"  (Currently using 2500 - consider lowering further)")

print(f"\nAvg Local Std Dev (NEW METRIC):")
print(f"  Brain avg:  {avg_local_std:.1f}")
print(f"  Face:       {face_stats['avg_local_std']:.1f}")
if face_stats['avg_local_std'] > avg_local_std * 1.5:
    print(f"  Recommendation: Use threshold ~{avg_local_std * 1.3:.1f}")

os.remove('temp_face.jpg')
