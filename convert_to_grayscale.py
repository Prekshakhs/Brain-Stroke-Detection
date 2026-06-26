# File: convert_to_grayscale.py

import cv2
import os
from glob import glob

def convert_dataset_to_grayscale(data_dir='data'):
    """Convert all colorized images to grayscale"""
    
    print("🔄 CONVERTING COLORIZED IMAGES TO GRAYSCALE")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        for class_name in ['NoStroke', 'Hemorrhagic', 'Ischemic']:
            path = os.path.join(data_dir, split, class_name)
            
            if not os.path.exists(path):
                continue
            
            images = glob(f"{path}/*.jpg") + glob(f"{path}/*.png")
            
            print(f"\n📁 {split}/{class_name}: {len(images)} images")
            
            converted = 0
            for img_path in images:
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    
                    # Check if already grayscale
                    if len(img.shape) == 2:
                        continue
                    
                    # Check if it's actually colored (not just 3-channel grayscale)
                    if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std() < img.std():
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Save as 3-channel for compatibility
                        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        
                        cv2.imwrite(img_path, gray_3ch)
                        converted += 1
                
                except Exception as e:
                    print(f"   ⚠️  Error processing {os.path.basename(img_path)}: {e}")
            
            if converted > 0:
                print(f"   ✅ Converted {converted} images to grayscale")
            else:
                print(f"   ℹ️  Already grayscale or minimal color")
    
    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    convert_dataset_to_grayscale('data')