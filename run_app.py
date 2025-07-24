#!/usr/bin/env python3
"""
Face Recognition Attendance System - Launcher
This script helps launch the Streamlit application with proper setup.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "Attendance",
        "StudentDetails", 
        "TrainingImage",
        "TrainingImageLabel"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("✅ Directory structure setup complete!")

def check_haarcascade():
    """Check if haarcascade file exists"""
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        print("⚠️  Haarcascade file not found!")
        print("Please download 'haarcascade_frontalface_default.xml' from:")
        print("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
        return False
    print("✅ Haarcascade file found!")
    return True

def run_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Starting Face Recognition Attendance System...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main setup and launch function"""
    print("🎓 Face Recognition Attendance System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check haarcascade file
    if not check_haarcascade():
        print("\n❌ Setup incomplete. Please download the required haarcascade file.")
        return
    
    print("\n✅ Setup complete!")
    print("🚀 Launching application...")
    print("\nThe application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    print("=" * 50)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()
