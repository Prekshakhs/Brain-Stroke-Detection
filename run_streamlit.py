# File: run_streamlit.py
import os
import sys
import subprocess

def run_streamlit_app():
    """Launch the Streamlit application"""
    
    # Change to streamlit_app directory
    app_dir = 'streamlit_app'
    if not os.path.exists(app_dir):
        print(f"❌ Directory {app_dir} not found!")
        return
    
    # Check if app.py exists
    app_file = os.path.join(app_dir, 'app.py')
    if not os.path.exists(app_file):
        print(f"❌ File {app_file} not found!")
        return
    
    print("🚀 Launching Streamlit Brain Stroke Detection App...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit_app()