@echo off
title Face Recognition Attendance System
color 0A

echo.
echo ====================================================
echo    Face Recognition Attendance System Launcher
echo ====================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [INFO] Python is installed
echo.

:: Create directories
if not exist "Attendance" mkdir "Attendance"
if not exist "StudentDetails" mkdir "StudentDetails"
if not exist "TrainingImage" mkdir "TrainingImage"
if not exist "TrainingImageLabel" mkdir "TrainingImageLabel"

echo [INFO] Directory structure created
echo.

:: Check for haarcascade file
if not exist "haarcascade_frontalface_default.xml" (
    echo [WARNING] haarcascade_frontalface_default.xml not found!
    echo Please download it from:
    echo https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        echo Please download the file and try again.
        pause
        exit /b 1
    )
)

:: Install requirements
echo [INFO] Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Setup complete!
echo [INFO] Starting the application...
echo.
echo The application will open in your default web browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo ====================================================
echo.

:: Run the application
streamlit run main.py

pause
