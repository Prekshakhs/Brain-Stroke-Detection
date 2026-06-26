#!/usr/bin/env python
"""
Verify that model files are accessible from the Streamlit app
"""
import os
import sys

print("=" * 60)
print("🔍 Model Directory Verification")
print("=" * 60)

# Check current working directory
cwd = os.getcwd()
print(f"\n📍 Current Working Directory: {cwd}")

# Possible paths the app might use
possible_paths = [
    "../models",
    "../../models",
    os.path.join(os.path.dirname(__file__), "models"),
    os.path.join(os.path.dirname(__file__), "streamlit_app", "..", "models"),
]

print(f"\n🔎 Checking possible model paths:")
for i, path in enumerate(possible_paths, 1):
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    status = "✅ EXISTS" if exists else "❌ NOT FOUND"
    print(f"\n   {i}. Path: {path}")
    print(f"      Absolute: {abs_path}")
    print(f"      Status: {status}")
    
    if exists and os.path.isdir(abs_path):
        files = [f for f in os.listdir(abs_path) if f.endswith(('.pth', '.ckpt', '.h5'))]
        if files:
            print(f"      Models found: {len(files)}")
            for f in files:
                print(f"         - {f}")
        else:
            print(f"      ⚠️ No model files (.pth, .ckpt, .h5) found")

# Direct check from streamlit_app directory
print(f"\n" + "=" * 60)
print("📂 Direct models directory check:")
print("=" * 60)

models_dir = os.path.join(os.path.dirname(__file__), "models")
print(f"\nAbsolute path to models: {models_dir}")

if os.path.exists(models_dir):
    print(f"✅ Models directory EXISTS")
    files = os.listdir(models_dir)
    print(f"📋 Contents: {len(files)} items")
    
    model_files = [f for f in files if f.endswith(('.pth', '.ckpt', '.h5'))]
    if model_files:
        print(f"\n✅ Found {len(model_files)} model file(s):")
        for f in model_files:
            file_path = os.path.join(models_dir, f)
            size = os.path.getsize(file_path) / (1024*1024)  # Convert to MB
            print(f"   ✓ {f} ({size:.2f} MB)")
    else:
        print(f"\n❌ No model files found in: {models_dir}")
        print(f"   Directory contents: {files}")
else:
    print(f"❌ Models directory DOES NOT EXIST: {models_dir}")

print("\n" + "=" * 60)
print("✅ Verification Complete")
print("=" * 60)
