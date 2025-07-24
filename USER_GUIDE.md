# 📖 User Guide - Face Recognition Attendance System

## 🚀 Getting Started

### Step 1: Launch the Application
Choose one of these methods to start the application:

**Option A: Using Batch File (Windows)**
- Double-click `run_app.bat`

**Option B: Using PowerShell (Windows)**
- Right-click `run_app.ps1` → Run with PowerShell

**Option C: Using Python Launcher**
```bash
python run_app.py
```

**Option D: Direct Streamlit**
```bash
streamlit run main.py
```

### Step 2: Access the Web Interface
- The application will automatically open in your default web browser
- If it doesn't open, go to: `http://localhost:8501`

## 👨‍💼 Admin Panel Guide

### Initial Setup
1. **First Login**: Use the default password "admin123" (or check `psd.txt`)
2. **Change Password**: Immediately change the default password for security

### Registering Students
1. Go to "Register Student" tab
2. Enter Student ID (unique identifier)
3. Enter Student Name (alphabets only)
4. Click "Capture Images"
5. Look at the camera for 50 photos to be taken
6. Wait for completion message

### Training the Model
1. Go to "Train Model" tab
2. Click "Start Training" (do this after registering students)
3. Wait for training to complete
4. You'll see a success message when done

### Managing Students
1. Go to "View Students" tab to see all registered students
2. View total count and student details

### Viewing Attendance
1. Go to "Attendance Records" tab
2. View today's attendance automatically
3. Select specific dates from dropdown
4. Download attendance reports as CSV files

### Email Configuration
1. Go to "Email Settings" tab
2. Enter system email address (e.g., Gmail)
3. Enter email app password (not regular password)
4. Configure SMTP settings (default: Gmail settings)
5. Click "Save Email Configuration"
6. Students can now receive attendance via email

### Changing Password
1. Go to "Change Password" tab
2. Enter current password
3. Enter new password
4. Confirm new password
5. Click "Change Password"

## 👨‍🎓 Student Panel Guide

### Student Login
1. Enter your Student ID (the same ID used during registration)
2. Click "Login as Student"
3. You'll see a welcome message with your name

### Taking Attendance
1. Go to "Take Attendance" tab
2. Click "Start Attendance Camera"
3. Look at the camera
4. Wait for your face to be recognized
5. You'll see a success message when attendance is marked

### Viewing Your Attendance
1. Go to "My Attendance" tab
2. View your complete attendance history
3. See statistics:
   - Total days present
   - This month's attendance
   - Today's status
4. Download your personal attendance record
5. Email your attendance (if email is configured)

### Setting Up Email
1. Go to "Email Settings" tab
2. Enter your email address
3. Click "Save Email Address"
4. Send test email to verify functionality
5. Use "Email My Attendance" button in "My Attendance" tab

## 🔧 Troubleshooting

### Camera Issues
- **Camera not working**: Check if another app is using the camera
- **Permission denied**: Allow camera access in browser/system settings
- **Wrong camera**: Modify `config.py` and change `CAMERA_INDEX`

### Recognition Issues
- **Face not detected**: Ensure good lighting and face the camera directly
- **Wrong person recognized**: Retrain the model with more training images
- **Unknown person**: Make sure the person is registered and model is trained

### Login Issues
- **Admin password wrong**: Check `TrainingImageLabel/psd.txt` file
- **Student ID not found**: Verify the ID exists in student records
- **Can't access panels**: Clear browser cache and restart application

### Performance Issues
- **Slow recognition**: Adjust confidence threshold in `config.py`
- **App running slowly**: Close other applications using camera/CPU
- **Training takes long**: This is normal for large numbers of students

## 📋 Best Practices

### For Administrators
1. **Regular Backups**: Backup the entire project folder regularly
2. **Password Security**: Use strong passwords and change them regularly
3. **Model Retraining**: Retrain the model when adding many new students
4. **Monitor Accuracy**: Check recognition accuracy and retrain if needed

### For Students
1. **Good Lighting**: Take attendance in well-lit areas
2. **Face the Camera**: Look directly at the camera for best results
3. **Remove Obstructions**: Remove glasses, masks, or hats if recognition fails
4. **One Person**: Ensure only one person is in the camera frame

### Data Management
1. **Regular Cleanup**: Archive old attendance files periodically
2. **Export Reports**: Regularly export attendance data for backup
3. **Check Storage**: Monitor disk space for image storage
4. **Update Records**: Keep student information updated

## 📊 Understanding the Data

### Attendance Files
- Located in `Attendance/` folder
- Named as `Attendance_DD-MM-YYYY.csv`
- Contains: ID, Name, Date, Time

### Student Details
- Located in `StudentDetails/StudentDetails.csv`
- Contains: Serial No, ID, Name

### Training Images
- Located in `TrainingImage/` folder
- Named as `Name.Serial.ID.ImageNumber.jpg`
- 50 images per student for training

## 🔄 Maintenance

### Weekly Tasks
- Check attendance accuracy
- Verify camera functionality
- Review system performance

### Monthly Tasks
- Backup all data
- Archive old attendance files
- Update student records if needed
- Change admin password (recommended)

### Semester/Term Tasks
- Complete data export
- Clean up old training images if needed
- Update system if new version available
- Performance review and optimization

## 📞 Getting Help

### Error Messages
- Read error messages carefully
- Check the troubleshooting section above
- Restart the application if issues persist

### Data Recovery
- Check backup folders for lost data
- Recreate student records if CSV is corrupted
- Retrain model if training files are lost

### Performance Optimization
- Close unnecessary browser tabs
- Restart the application periodically
- Check system resources (CPU, memory)
- Update camera drivers if needed

---

**Need more help?** Check the README.md file or contact your system administrator.
