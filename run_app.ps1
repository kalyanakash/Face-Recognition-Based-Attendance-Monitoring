# Face Recognition Attendance System Launcher for Windows
# PowerShell Script

Write-Host "🎓 Face Recognition Attendance System" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python not found! Please install Python 3.7+ first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    pause
    exit 1
}

# Check if pip is available
try {
    pip --version > $null 2>&1
    Write-Host "✅ pip is available" -ForegroundColor Green
}
catch {
    Write-Host "❌ pip not found! Please install pip first." -ForegroundColor Red
    pause
    exit 1
}

# Create directories if they don't exist
$directories = @("Attendance", "StudentDetails", "TrainingImage", "TrainingImageLabel")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir
        Write-Host "📁 Created directory: $dir" -ForegroundColor Green
    }
}

# Check for haarcascade file
if (!(Test-Path "haarcascade_frontalface_default.xml")) {
    Write-Host "⚠️  Haarcascade file not found!" -ForegroundColor Yellow
    Write-Host "Please download 'haarcascade_frontalface_default.xml' from:" -ForegroundColor Yellow
    Write-Host "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml" -ForegroundColor Cyan
    Write-Host ""
    $download = Read-Host "Would you like to continue without it? (y/N)"
    if ($download -ne "y" -and $download -ne "Y") {
        Write-Host "Please download the file and run the script again." -ForegroundColor Yellow
        pause
        exit 1
    }
}

# Install requirements
Write-Host "📦 Installing required packages..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Error installing packages. Please check your internet connection." -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "🚀 Starting Face Recognition Attendance System..." -ForegroundColor Cyan
Write-Host ""
Write-Host "The application will open in your default web browser." -ForegroundColor Yellow
Write-Host "If it doesn't open automatically, go to: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Gray
Write-Host "=" * 50 -ForegroundColor Gray

# Run the Streamlit application
try {
    streamlit run main.py
}
catch {
    Write-Host "❌ Error running the application." -ForegroundColor Red
    Write-Host "Please ensure all requirements are installed correctly." -ForegroundColor Yellow
    pause
}
