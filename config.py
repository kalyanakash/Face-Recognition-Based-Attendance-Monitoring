# Face Recognition Attendance System Configuration

# Application Settings
APP_TITLE = "Face Recognition Attendance System"
APP_ICON = "🎓"

# Default Admin Credentials
DEFAULT_ADMIN_PASSWORD = "admin123"

# Face Recognition Settings
CONFIDENCE_THRESHOLD = 70  # Lower values mean stricter matching
FACE_DETECTION_SCALE_FACTOR = 1.2
FACE_DETECTION_MIN_NEIGHBORS = 5

# Image Capture Settings
TRAINING_IMAGES_COUNT = 50
CAMERA_INDEX = 0  # 0 for default camera, change if needed

# File Paths
HAARCASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_PATH = "TrainingImageLabel/Trainner.yml"
PASSWORD_FILE = "TrainingImageLabel/psd.txt"
STUDENT_DETAILS_PATH = "StudentDetails/StudentDetails.csv"

# Directory Paths
ATTENDANCE_DIR = "Attendance/"
STUDENT_DETAILS_DIR = "StudentDetails/"
TRAINING_IMAGE_DIR = "TrainingImage/"
TRAINING_LABEL_DIR = "TrainingImageLabel/"

# UI Settings
SIDEBAR_STATE = "expanded"
LAYOUT = "wide"

# Date Format
DATE_FORMAT = "%d-%m-%Y"
TIME_FORMAT = "%H:%M:%S"

# Email Configuration
# Note: Configure these settings for your email provider
EMAIL_SETTINGS = {
    "smtp_server": "smtp.gmail.com",  # Gmail SMTP server
    "smtp_port": 587,
    "use_tls": True,
    "sender_email": "",  # Will be configured by admin
    "sender_password": "",  # Will be configured by admin (use app password for Gmail)
    "sender_name": "Attendance System"
}

# Email Templates
EMAIL_TEMPLATES = {
    "subject": "Your Attendance Records - {date}",
    "body": """
Dear {student_name},

Please find your attendance records attached to this email.

Attendance Summary:
- Total Days Present: {total_days}
- This Month: {this_month}
- Today's Status: {today_status}

Best regards,
Attendance Management System
"""
}
