# File: src/data_utils.py
import os
import shutil
import random
from collections import defaultdict

class DatasetManager:
    def __init__(self, base_path='data'):
        self.base_path = base_path
        self.class_names = ['NoStroke', 'Hemorrhagic', 'Ischemic']
        
    def verify_dataset(self):
        """Verify dataset structure and count images"""
        print("🔍 Verifying dataset structure...")
        
        splits = ['train', 'val', 'test']
        total_images = 0
        
        for split in splits:
            print(f"\n📂 {split.upper()} SET:")
            split_total = 0
            
            for class_name in self.class_names:
                class_path = os.path.join(self.base_path, split, class_name)
                
                if os.path.exists(class_path):
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                    count = len(image_files)
                    split_total += count
                    print(f"   ✅ {class_name}: {count} images")
                else:
                    print(f"   ❌ {class_name}: folder missing")
            
            print(f"   📊 Total {split}: {split_total} images")
            total_images += split_total
        
        print(f"\n🎯 TOTAL DATASET: {total_images} images")
        return total_images > 0
    
    def split_dataset(self, source_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test"""
        print("📂 Splitting dataset...")
        
        if not os.path.exists(source_folder):
            print(f"❌ Source folder not found: {source_folder}")
            return
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            print(f"⚠️ Warning: Ratios sum to {total_ratio}, normalizing...")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        for class_name in self.class_names:
            source_class_path = os.path.join(source_folder, class_name)
            
            if not os.path.exists(source_class_path):
                print(f"⚠️ Class folder not found: {source_class_path}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(source_class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                print(f"⚠️ No images found in {source_class_path}")
                continue
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            total_images = len(image_files)
            train_end = int(total_images * train_ratio)
            val_end = train_end + int(total_images * val_ratio)
            
            # Split files
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Copy files to respective folders
            splits_data = [
                ('train', train_files),
                ('val', val_files),
                ('test', test_files)
            ]
            
            for split_name, files in splits_data:
                target_dir = os.path.join(self.base_path, split_name, class_name)
                os.makedirs(target_dir, exist_ok=True)
                
                for file in files:
                    source_file = os.path.join(source_class_path, file)
                    target_file = os.path.join(target_dir, file)
                    shutil.copy2(source_file, target_file)
            
            print(f"✅ {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        print("✅ Dataset splitting completed!")

# Usage example
if __name__ == "__main__":
    dataset_manager = DatasetManager()
    
    # If you have unsplit data in a single folder, use this:
    # dataset_manager.split_dataset('path/to/your/original/dataset')
    
    # Verify your dataset
    dataset_manager.verify_dataset()