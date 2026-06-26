#!/usr/bin/env python
"""
Quick launcher for the Grad-CAM analysis notebook
Handles setup and opens the notebook in your default environment
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'jupyter',
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'opencv-python',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main launcher"""
    print("=" * 60)
    print("🧠 Brain Stroke Detection - Grad-CAM Notebook Launcher")
    print("=" * 60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n📥 Installing missing packages...")
        for package in missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=False)
        print("✅ Packages installed!")
    else:
        print("✅ All dependencies available!")
    
    # Verify notebook exists
    notebook_path = Path("notebooks/model_analysis_gradcam.ipynb")
    
    if not notebook_path.exists():
        print(f"\n❌ Notebook not found: {notebook_path}")
        print("❌ Please ensure you're running from the project root directory")
        return
    
    print(f"\n📓 Found notebook: {notebook_path}")
    
    # Verify results directory
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"\n📁 Creating results directory...")
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Results directory created")
    
    # Launch options
    print("\n" + "=" * 60)
    print("🚀 Choose how to launch the notebook:")
    print("=" * 60)
    print("1. Jupyter Notebook (browser-based)")
    print("2. JupyterLab (advanced)")
    print("3. VS Code (with Jupyter extension)")
    print("4. Open file browser only")
    print("0. Cancel")
    
    choice = input("\nEnter your choice (0-4): ").strip()
    
    if choice == "1":
        print("\n📂 Launching Jupyter Notebook...")
        subprocess.Popen([
            sys.executable, '-m', 'jupyter', 'notebook',
            str(notebook_path)
        ])
        print("✅ Jupyter Notebook opened in your browser!")
        
    elif choice == "2":
        print("\n📂 Launching JupyterLab...")
        subprocess.Popen([
            sys.executable, '-m', 'jupyter', 'lab',
            str(notebook_path)
        ])
        print("✅ JupyterLab opened in your browser!")
        
    elif choice == "3":
        print("\n📂 Opening in VS Code...")
        subprocess.Popen([
            'code',
            str(notebook_path)
        ])
        print("✅ File opened in VS Code!")
        
    elif choice == "4":
        print(f"\n📁 Opening file location: {notebook_path.parent}")
        if sys.platform == 'win32':
            os.startfile(notebook_path.parent)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(notebook_path.parent)])
        else:
            subprocess.Popen(['xdg-open', str(notebook_path.parent)])
        
    elif choice == "0":
        print("\n👋 Cancelled")
        return
    else:
        print(f"\n❌ Invalid choice: {choice}")
        return
    
    print("\n" + "=" * 60)
    print("📖 NOTEBOOK FEATURES:")
    print("=" * 60)
    print("✨ Grad-CAM Visualization - Show model focus areas")
    print("👥 Patient Aggregation - Group results by patient ID")
    print("📊 Per-Patient Metrics - Calculate patient-level metrics")
    print("📈 Confusion Matrices - Visual performance analysis")
    print("💾 CSV Export - Download results for further analysis")
    print("📄 JSON Report - Comprehensive analysis report")
    print("\n" + "=" * 60)
    print("📚 For detailed guide, see: NOTEBOOK_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
