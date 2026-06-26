"""Test colorized brain heatmap acceptance"""
import cv2
import numpy as np

def is_brain_scan(image_array):
    """Updated balanced validation"""
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
            
            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    r_val = r[i, j]
                    g_val = g[i, j]
                    b_val = b[i, j]
                    
                    if (50 < r_val < 220 and 40 < g_val < 200 and 20 < b_val < 180 and
                        r_val > g_val > b_val):
                        skin_pixels += 1
            
            skin_ratio = skin_pixels / total_pixels
            
            if skin_ratio > 0.3:
                return False, "Skin tones detected"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)

# Create a colorized brain heatmap like the one in the image
print("="*60)
print("TESTING COLORIZED BRAIN HEATMAP")
print("="*60)

# Create a heatmap-like image with blue, cyan, yellow, red colors
heatmap = np.zeros((512, 512, 3), dtype=np.uint8)

# Create brain-like structure in center
center_x, center_y = 256, 256
for i in range(512):
    for j in range(512):
        dist_from_center = np.sqrt((i - center_y)**2 + (j - center_x)**2)
        
        # Brain-like circular/oval region
        if dist_from_center < 180:
            # Heatmap colors: blue -> cyan -> yellow -> red
            # Use distance to create a gradient
            intensity = (180 - dist_from_center) / 180.0
            
            if intensity < 0.33:
                # Blue region
                heatmap[i, j] = [255, 0, int(255 * (1 - intensity * 3))]
            elif intensity < 0.66:
                # Cyan to yellow
                t = (intensity - 0.33) / 0.33
                heatmap[i, j] = [255, int(255 * t), int(255 * (1 - t))]
            else:
                # Yellow to red
                t = (intensity - 0.66) / 0.34
                heatmap[i, j] = [int(255 * (1 - t)), 255, 0]
        else:
            # Background - dark blue
            heatmap[i, j] = [50, 0, 100]

# Add brain outlines with dark red
cv2.circle(heatmap, (256, 256), 180, (0, 0, 150), 3)
cv2.circle(heatmap, (256, 256), 175, (0, 20, 150), 2)

# Test it
result, msg = is_brain_scan(heatmap)
print(f"\nColorized heatmap test:")
print(f"Result: {'✅ ACCEPT' if result else '❌ REJECT'}")
print(f"Message: {msg}")

if result:
    print("\n✅ Colorized brain images are now ACCEPTED!")
else:
    print("\n❌ Issue: Colorized brain image rejected")

# Analyze its characteristics
print("\n" + "="*60)
print("HEATMAP CHARACTERISTICS")
print("="*60)
gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)

print(f"Unique gray values: {len(np.unique(gray))}")
print(f"Mean saturation: {np.mean(hsv[:,:,1]):.1f}")
print(f"Max saturation: {np.max(hsv[:,:,1])}")
print(f"Std saturation: {np.std(hsv[:,:,1]):.1f}")

edges = cv2.Canny(gray, 50, 150)
edge_ratio = np.sum(edges > 0) / (512 * 512)
print(f"Edge ratio: {edge_ratio:.4f}")

# Check skin tones
r = heatmap[:,:,2].astype(float)
g = heatmap[:,:,1].astype(float)
b = heatmap[:,:,0].astype(float)

skin_pixels = 0
for i in range(512):
    for j in range(512):
        r_val = r[i, j]
        g_val = g[i, j]
        b_val = b[i, j]
        
        if (50 < r_val < 220 and 40 < g_val < 200 and 20 < b_val < 180 and
            r_val > g_val > b_val):
            skin_pixels += 1

skin_ratio = skin_pixels / (512 * 512)
print(f"Skin tone ratio: {skin_ratio:.4f}")
