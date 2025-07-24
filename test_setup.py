"""
Test script for Face Recognition Attendance System
This script performs basic checks to ensure the system is set up correctly.
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"✅ {package_name}: Installed")
            return True
        else:
            print(f"❌ {package_name}: Not installed")
            return False
    except Exception as e:
        print(f"❌ {package_name}: Error checking - {e}")
        return False

def main():
    print("🎓 Face Recognition Attendance System - Health Check")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} (Requires 3.7+)")
    
    print("\n📁 File Structure Check:")
    print("-" * 30)
    
    # Check main files
    files_to_check = [
        ("main.py", "Main application file"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "Documentation"),
        ("USER_GUIDE.md", "User guide"),
        ("config.py", "Configuration file"),
        ("haarcascade_frontalface_default.xml", "Face detection model")
    ]
    
    all_files_ok = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_files_ok = False
    
    # Check directories
    print("\n📂 Directory Structure Check:")
    print("-" * 30)
    
    directories_to_check = [
        ("Attendance", "Attendance records"),
        ("StudentDetails", "Student information"),
        ("TrainingImage", "Training images"),
        ("TrainingImageLabel", "Trained models")
    ]
    
    all_dirs_ok = True
    for dirpath, description in directories_to_check:
        if not check_directory_exists(dirpath, description):
            all_dirs_ok = False
    
    # Check Python packages
    print("\n📦 Python Packages Check:")
    print("-" * 30)
    
    packages_to_check = [
        "streamlit",
        "cv2",
        "numpy",
        "PIL",
        "pandas"
    ]
    
    all_packages_ok = True
    for package in packages_to_check:
        if not check_python_package(package):
            all_packages_ok = False
    
    # Check student data
    print("\n👥 Data Check:")
    print("-" * 30)
    
    student_file = "StudentDetails/StudentDetails.csv"
    if os.path.exists(student_file):
        try:
            import pandas as pd
            df = pd.read_csv(student_file)
            student_count = len(df) - 1  # Subtract header row
            print(f"✅ Student Records: {student_count} students registered")
        except Exception as e:
            print(f"⚠️  Student Records: Error reading file - {e}")
    else:
        print("ℹ️  Student Records: No students registered yet")
    
    # Check training model
    trainer_file = "TrainingImageLabel/Trainner.yml"
    if os.path.exists(trainer_file):
        print("✅ Training Model: Model exists")
    else:
        print("ℹ️  Training Model: No trained model found")
    
    # Check password file
    password_file = "TrainingImageLabel/psd.txt"
    if os.path.exists(password_file):
        print("✅ Admin Password: Password file exists")
    else:
        print("ℹ️  Admin Password: Using default password")
    
    # Summary
    print("\n" + "=" * 60)
    if all_files_ok and all_dirs_ok and all_packages_ok:
        print("🎉 System Status: READY TO USE!")
        print("✅ All components are properly set up.")
        print("\nNext steps:")
        print("1. Run the application: streamlit run main.py")
        print("2. Login as admin and register students")
        print("3. Train the face recognition model")
        print("4. Start taking attendance!")
    else:
        print("⚠️  System Status: SETUP REQUIRED")
        print("❌ Some components are missing or not properly configured.")
        print("\nPlease fix the issues above before running the application.")
        
        if not all_packages_ok:
            print("\nTo install missing packages:")
            print("pip install -r requirements.txt")
        
        if not os.path.exists("haarcascade_frontalface_default.xml"):
            print("\nTo download haarcascade file:")
            print("Visit: https://github.com/opencv/opencv/blob/master/data/haarcascades/")
    
    print("\n📖 For detailed instructions, see USER_GUIDE.md")

if __name__ == "__main__":
    main()
