# 🎓 Face Recognition Based Attendance Monitoring System

A modern, web-based attendance system using face recognition technology built with Streamlit. The system features separate admin and student panels for comprehensive attendance management.

## ✨ Features

### 👨‍💼 Admin Panel
- **Secure Login**: Password-protected admin access
- **Student Registration**: Register new students with face capture
- **Model Training**: Train face recognition models
- **Student Management**: View all registered students
- **Attendance Records**: View and download attendance reports
- **Password Management**: Change admin password

### 👨‍🎓 Student Panel
- **Student Login**: ID-based student access
- **Take Attendance**: Mark attendance using face recognition
- **Personal Records**: View individual attendance history
- **Download Reports**: Export personal attendance data
- **Attendance Statistics**: Track daily and monthly attendance

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
python run_app.py
```

### Method 2: Manual Setup
1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Haarcascade File**
   Download `haarcascade_frontalface_default.xml` from OpenCV repository and place it in the project root.

3. **Run the Application**
   ```bash
   streamlit run main.py
   ```

## 📋 Requirements

- Python 3.7+
- Webcam/Camera
- Required packages (see requirements.txt):
  - streamlit
  - opencv-contrib-python
  - numpy
  - pillow
  - pandas
  - python-dateutil

## 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kalyanakash/Face-Recognition-Based-Attendance-Monitoring.git
   cd Face-Recognition-Based-Attendance-Monitoring-System
   ```

2. **Set up Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Files**
   - Download `haarcascade_frontalface_default.xml` from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - Place it in the project root directory

5. **Run the Application**
   ```bash
   streamlit run main.py
   ```

## 🎯 Usage Guide

### First Time Setup
1. Start the application
2. Login as Admin (default password: check `psd.txt` or use "admin123")
3. Register students in the Admin Panel
4. Train the face recognition model
5. Students can now login and take attendance

### Admin Workflow
1. **Login**: Use admin password to access admin panel
2. **Register Students**: 
   - Enter Student ID and Name
   - Capture 50 face images using webcam
3. **Train Model**: Train the face recognition model after registering students
4. **Monitor**: View attendance records and manage students

### Student Workflow
1. **Login**: Enter your Student ID to access student panel
2. **Take Attendance**: Use webcam to mark attendance via face recognition
3. **View Records**: Check your personal attendance history
4. **Download Data**: Export your attendance records

## 📁 Project Structure

```
Face-Recognition-Based-Attendance-Monitoring-System/
├── main.py                              # Main Streamlit application
├── run_app.py                          # Application launcher
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
├── haarcascade_frontalface_default.xml # Face detection model
├── Attendance/                         # Attendance CSV files
├── StudentDetails/                     # Student information
├── TrainingImage/                      # Training images
└── TrainingImageLabel/                 # Trained models and passwords
```

## 🔐 Security Features

- **Admin Authentication**: Secure password-based admin access
- **Student Verification**: ID-based student authentication
- **Data Protection**: Local storage of sensitive information
- **Session Management**: Secure login sessions

## 📊 Data Management

### Student Data
- Student details stored in CSV format
- Face images organized by ID and name
- Trained models saved locally

### Attendance Data
- Daily attendance files with timestamps
- Duplicate prevention for same-day attendance
- CSV export functionality

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check camera permissions
   - Ensure no other application is using the camera
   - Try different camera index (modify cv2.VideoCapture(0) to cv2.VideoCapture(1))

2. **Model Training Fails**
   - Ensure students are registered first
   - Check if training images exist in TrainingImage folder
   - Verify haarcascade file is present

3. **Face Not Recognized**
   - Ensure good lighting conditions
   - Re-train the model after adding new students
   - Check if the person is registered in the system

4. **Login Issues**
   - Check admin password in TrainingImageLabel/psd.txt
   - Verify student ID exists in StudentDetails.csv

## 🔄 Updates and Improvements

### Version 2.0 Features
- ✅ Streamlit web interface
- ✅ Separate admin and student panels
- ✅ Real-time face recognition
- ✅ Enhanced UI/UX
- ✅ Session management
- ✅ Data export functionality
- ✅ Attendance statistics

### Future Enhancements
- [ ] Email notifications
- [ ] Database integration
- [ ] Mobile app support
- [ ] Advanced analytics
- [ ] Multi-camera support

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for face recognition capabilities
- Streamlit for the web framework
- Contributors and users of the system

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team

---

Made with ❤️ for educational institutions and organizations needing efficient attendance management.
A python GUI integrated attendance system using face recognition to take attendance.

In this python project, I have made an attendance system which takes attendance by using face recognition technique. I have also intergrated it with GUI (Graphical user interface) so it can be easy to use by anyone. GUI for this project is also made on python using tkinter.

TECHNOLOGY USED:
1) tkinter for whole GUI
2) OpenCV for taking images and face recognition (cv2.face.LBPHFaceRecognizer_create())
3) CSV, Numpy, Pandas, datetime etc. for other purposes.

FEATURES:
1) Easy to use with interactive GUI support.
2) Password protection for new person registration.
3) Creates/Updates CSV file for deatils of students on registration.
4) Creates a new CSV file everyday for attendance and marks attendance with proper date and time.
5) Displays live attendance updates for the day on the main screen in tabular format with Id, name, date and time.

# SCREENSHOTS
MAIN SCREEN:
![Screenshot (9)](https://user-images.githubusercontent.com/37211676/58502148-97ec2a00-81a3-11e9-963e-674b9c3e05dc.png)

TAKING ATTENDANCE:
![Screenshot (10)](https://user-images.githubusercontent.com/37211676/58502149-97ec2a00-81a3-11e9-9658-8968da396c2e.png)

SHOWING ATTENDANCE TAKEN:
![Screenshot (11)](https://user-images.githubusercontent.com/37211676/58502151-9884c080-81a3-11e9-9a90-fec29940ee5a.png)

HELP OPTION IN MENUBAR:
![Screenshot (12)](https://user-images.githubusercontent.com/37211676/58502152-991d5700-81a3-11e9-861a-9115526010c2.png)

CHANGE PASSWORD OPTION:
![Screenshot (13)](https://user-images.githubusercontent.com/37211676/58502146-97539380-81a3-11e9-8536-0c68160ecc55.png)
