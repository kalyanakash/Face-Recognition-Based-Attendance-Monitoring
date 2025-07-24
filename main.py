# Enhanced Streamlit Face Recognition Attendance System with Admin & User Panels
import streamlit as st
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import io
import base64

# Function to set background image
def set_background_image():
    """Set background image for the Streamlit app"""
    # Background image removed - using clean default background
    background_css = f"""
    <style>
    .stApp {{
        background-color: #f0f2f6;
    }}
    
    /* Make content areas clean and readable */
    .main .block-container {{
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    /* Style for login cards */
    .stColumns > div > div {{
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }}
    
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 0 2px;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }}
    
    /* Header styling */
    h1, h2, h3 {{
        color: #1e3d59;
    }}
    </style>
    """
    
    st.markdown(background_css, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    if 'current_user_name' not in st.session_state:
        st.session_state.current_user_name = None

# Authentication functions
def get_admin_password():
    """Get admin password from file"""
    try:
        with open("TrainingImageLabel/psd.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "admin123"  # Default password

def verify_user_credentials(user_id):
    """Verify if user ID exists in the system"""
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        user_exists = any(df['ID'].astype(str) == str(user_id))
        if user_exists:
            user_name = df.loc[df['ID'].astype(str) == str(user_id), 'NAME'].iloc[0]
            return True, user_name
        return False, None
    except FileNotFoundError:
        return False, None

def logout():
    """Logout function"""
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.current_user_id = None
    st.session_state.current_user_name = None
    st.rerun()

# Email Configuration Functions
def get_email_config():
    """Get email configuration from file"""
    config_file = "TrainingImageLabel/email_config.txt"
    try:
        with open(config_file, "r") as f:
            lines = f.readlines()
            config = {}
            for line in lines:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value
            return config
    except FileNotFoundError:
        return {}

def save_email_config(sender_email, sender_password, smtp_server="smtp.gmail.com", smtp_port="587"):
    """Save email configuration to file"""
    assure_path_exists("TrainingImageLabel/")
    config_file = "TrainingImageLabel/email_config.txt"
    with open(config_file, "w") as f:
        f.write(f"sender_email={sender_email}\n")
        f.write(f"sender_password={sender_password}\n")
        f.write(f"smtp_server={smtp_server}\n")
        f.write(f"smtp_port={smtp_port}\n")

def send_attendance_email(recipient_email, student_name, student_id, attendance_data):
    """Send attendance data via email"""
    try:
        # Get email configuration
        email_config = get_email_config()
        
        if not email_config.get('sender_email') or not email_config.get('sender_password'):
            st.error("Email not configured! Please contact admin to set up email settings.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = f"Attendance Records - {student_name} ({student_id})"
        
        # Create email body
        total_days = len(attendance_data)
        current_month = datetime.datetime.now().strftime('%m-%Y')
        monthly_attendance = [record for record in attendance_data 
                            if current_month in record['Date']]
        today = datetime.datetime.now().strftime('%d-%m-%Y')
        today_attendance = [record for record in attendance_data 
                          if record['Date'] == today]
        today_status = "Present" if today_attendance else "Absent"
        
        body = f"""
Dear {student_name},

Please find your attendance records attached to this email.

Attendance Summary:
- Student ID: {student_id}
- Total Days Present: {total_days}
- This Month: {len(monthly_attendance)}
- Today's Status: {today_status}

The detailed attendance records are attached as a CSV file.

Best regards,
Face Recognition Attendance System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Create CSV attachment
        df = pd.DataFrame(attendance_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Attach CSV file
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(csv_data.encode())
        encoders.encode_base64(attachment)
        attachment.add_header(
            'Content-Disposition',
            f'attachment; filename=attendance_{student_name}_{student_id}.csv'
        )
        msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP(email_config.get('smtp_server', 'smtp.gmail.com'), 
                             int(email_config.get('smtp_port', 587)))
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        text = msg.as_string()
        server.sendmail(email_config['sender_email'], recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def get_student_email(student_id):
    """Get student email from StudentDetails CSV"""
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        # Check if EMAIL column exists, if not return None
        if 'EMAIL' in df.columns:
            email_row = df[df['ID'].astype(str) == str(student_id)]
            if not email_row.empty and pd.notna(email_row['EMAIL'].iloc[0]):
                return email_row['EMAIL'].iloc[0]
    except:
        pass
    return None

def update_student_email(student_id, email):
    """Update student email in StudentDetails CSV"""
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        
        # Add EMAIL column if it doesn't exist
        if 'EMAIL' not in df.columns:
            df['EMAIL'] = ''
        
        # Update email for the student
        df.loc[df['ID'].astype(str) == str(student_id), 'EMAIL'] = email
        df.to_csv("StudentDetails/StudentDetails.csv", index=False)
        return True
    except Exception as e:
        st.error(f"Failed to update email: {str(e)}")
        return False

# Directory setup
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if not exists:
        st.error("Haarcascade file missing. Please upload haarcascade_frontalface_default.xml.")
        st.stop()

# Save password logic
def save_password(old_pass, new_pass, confirm_pass):
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        with open("TrainingImageLabel/psd.txt", "r") as tf:
            key = tf.read()
    else:
        st.warning("No old password found. Registering new password.")
        with open("TrainingImageLabel/psd.txt", "w") as tf:
            tf.write(new_pass)
        st.success("Password registered successfully.")
        return

    if old_pass == key:
        if new_pass == confirm_pass:
            with open("TrainingImageLabel/psd.txt", "w") as tf:
                tf.write(new_pass)
            st.success("Password changed successfully.")
        else:
            st.error("New passwords do not match.")
    else:
        st.error("Old password is incorrect.")

# Train function
def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        st.error('Please register someone first.')
        return
    recognizer.save("TrainingImageLabel/Trainner.yml")
    st.success("Profile trained successfully.")

# Image loader for training
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

# Enhanced take images function with Streamlit camera
def TakeImages(Id, name, email=None):
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    
    # Check if user already exists
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        if any(df['ID'].astype(str) == str(Id)):
            st.error(f"ID {Id} already exists in the system!")
            return False
    
    serial = 1
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME', 'EMAIL']

    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            rows = list(reader1)
            serial = len(rows) // 2 if len(rows) > 1 else 1
            
        # Check if EMAIL column exists, if not add it
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        if 'EMAIL' not in df.columns:
            df['EMAIL'] = ''
            df.to_csv("StudentDetails/StudentDetails.csv", index=False)
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)

    if name.replace(" ", "").isalpha():
        st.info("Starting camera... Please look at the camera and wait for 50 photos to be taken.")
        
        # Create placeholders for camera feed and progress
        camera_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        cam = cv2.VideoCapture(0)
        
        # Set camera properties for better quality
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)    # Set width to 1280
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # Set height to 720
        cam.set(cv2.CAP_PROP_FPS, 30)              # Set FPS to 30
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Enable autofocus
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # Enable auto exposure
        cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)      # Adjust brightness
        cam.set(cv2.CAP_PROP_CONTRAST, 0.5)        # Adjust contrast
        cam.set(cv2.CAP_PROP_SATURATION, 0.5)      # Adjust saturation
        
        # Wait for camera to warm up
        time.sleep(2)
        
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        sampleNum = 0
        
        while sampleNum < 50:
            ret, img = cam.read()
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Apply image enhancement for better quality
            img = cv2.bilateralFilter(img, 9, 75, 75)  # Noise reduction while preserving edges
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Only capture if face is large enough (good distance from camera)
                if w > 100 and h > 100:
                    sampleNum += 1
                    
                    # Extract and enhance the face region
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (200, 200))  # Standardize face size
                    
                    cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg", face_roi)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"Sample: {sampleNum}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Quality: Good", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Face too small - show warning
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(img, "Move closer", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Convert BGR to RGB for Streamlit
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(img_rgb, channels="RGB", caption=f"Capturing images... {sampleNum}/50")
            
            # Update progress
            progress_bar.progress(sampleNum / 50)
            status_text.text(f"Images captured: {sampleNum}/50")
            
            if sampleNum >= 50:
                break
                
        cam.release()
        cv2.destroyAllWindows()

        # Save student details with email
        try:
            df = pd.read_csv('StudentDetails/StudentDetails.csv')
            if 'EMAIL' not in df.columns:
                df['EMAIL'] = ''
            
            new_row = pd.DataFrame({
                'SERIAL NO.': [serial],
                '': [''],
                'ID': [Id],
                '.1': [''],
                'NAME': [name],
                'EMAIL': [email if email else '']
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv('StudentDetails/StudentDetails.csv', index=False)
        except:
            # Fallback to original method
            with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([serial, '', Id, '', name, email if email else ''])

        st.success(f"Successfully captured 50 images for {name} (ID: {Id})")
        if email:
            st.success(f"Email {email} registered for student")
        st.info("Please train the model now to enable face recognition for this user.")
        return True
    else:
        st.error("Name must contain only alphabets")
        return False

# Enhanced attendance tracking with real-time detection
def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    
    # Check if trainer file exists
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        st.error("No trained model found! Please train the model first in Admin Panel.")
        return
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Load student details
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    except FileNotFoundError:
        st.error("No student details found! Please register students first.")
        return

    cam = cv2.VideoCapture(0)
    
    # Set camera properties for better quality during attendance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)    # Set width to 1280
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    # Set height to 720
    cam.set(cv2.CAP_PROP_FPS, 30)              # Set FPS to 30
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Enable autofocus
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # Enable auto exposure
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)      # Adjust brightness
    cam.set(cv2.CAP_PROP_CONTRAST, 0.5)        # Adjust contrast
    
    # Wait for camera to warm up
    time.sleep(2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance_data = []
    recognized_ids = set()  # To avoid duplicate entries
    
    # Streamlit placeholders
    camera_placeholder = st.empty()
    attendance_placeholder = st.empty()
    stop_button = st.button("Stop Attendance", key="stop_attendance")
    
    st.info("Camera started. Press 'Stop Attendance' button to finish.")
    
    frame_count = 0
    while not stop_button:
        ret, im = cam.read()
        if not ret:
            st.error("Failed to access camera")
            break
        
        frame_count += 1
        
        # Apply image enhancement for better recognition
        im = cv2.bilateralFilter(im, 9, 75, 75)  # Noise reduction
        
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            # Only process faces that are large enough for good recognition
            if w > 80 and h > 80:
                # Extract and resize face region for better recognition
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                serial, conf = recognizer.predict(face_roi)
                
                if conf < 70:  # Confidence threshold
                    try:
                        name = df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0]
                        ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values[0]
                        
                        # Add to attendance if not already marked
                        if ID not in recognized_ids:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance_data.append([ID, name, date, timeStamp])
                            recognized_ids.add(ID)
                            st.success(f"Attendance marked for {name} (ID: {ID})")
                        
                        cv2.putText(im, f"{name} (ID: {ID})", (x, y + h + 30), font, 0.7, (0, 255, 0), 2)
                        cv2.putText(im, f"Conf: {int(100-conf)}%", (x, y - 10), font, 0.5, (0, 255, 0), 1)
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except:
                        cv2.putText(im, "Unknown", (x, y + h + 30), font, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y + h + 30), font, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                # Face too small - show warning
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(im, "Move closer", (x, y-10), font, 0.6, (255, 255, 0), 2)
        
        # Convert BGR to RGB for Streamlit
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(im_rgb, channels="RGB", caption="Live Attendance Tracking")
        
        # Show current attendance
        if attendance_data:
            df_current = pd.DataFrame(attendance_data, columns=['ID', 'Name', 'Date', 'Time'])
            attendance_placeholder.dataframe(df_current, use_container_width=True)
        
        # Break if stop button is pressed
        if frame_count % 10 == 0:  # Check stop button every 10 frames
            time.sleep(0.1)

    cam.release()
    cv2.destroyAllWindows()

    # Save attendance data
    if attendance_data:
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        filename = f"Attendance/Attendance_{date}.csv"
        df_attendance = pd.DataFrame(attendance_data, columns=['ID', 'Name', 'Date', 'Time'])
        
        # Check if file exists and append or create new
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            # Remove duplicates based on ID and Date
            combined_df = pd.concat([existing_df, df_attendance])
            combined_df = combined_df.drop_duplicates(subset=['ID', 'Date'], keep='last')
            combined_df.to_csv(filename, index=False)
        else:
            df_attendance.to_csv(filename, index=False)
        
        st.success(f"Attendance saved for {len(attendance_data)} students!")
        st.dataframe(df_attendance, use_container_width=True)
    else:
        st.warning("No attendance was recorded.")

# Function to get user's attendance history
def get_user_attendance(user_id):
    """Get attendance history for a specific user"""
    attendance_files = []
    attendance_folder = "Attendance/"
    
    if os.path.exists(attendance_folder):
        for file in os.listdir(attendance_folder):
            if file.startswith("Attendance_") and file.endswith(".csv"):
                attendance_files.append(os.path.join(attendance_folder, file))
    
    user_attendance = []
    for file in attendance_files:
        try:
            df = pd.read_csv(file)
            user_records = df[df['ID'].astype(str) == str(user_id)]
            if not user_records.empty:
                user_attendance.extend(user_records.to_dict('records'))
        except:
            continue
    
    return user_attendance

# Function to display all students
def show_all_students():
    """Display all registered students"""
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        if not df.empty:
            # Clean the dataframe and include email if available
            if 'EMAIL' in df.columns:
                clean_df = df[['SERIAL NO.', 'ID', 'NAME', 'EMAIL']].dropna(subset=['ID', 'NAME'])
            else:
                clean_df = df[['SERIAL NO.', 'ID', 'NAME']].dropna(subset=['ID', 'NAME'])
            st.dataframe(clean_df, use_container_width=True)
            return len(clean_df)
        else:
            st.info("No students registered yet.")
            return 0
    except FileNotFoundError:
        st.info("No student database found.")
        return 0

# Function to show today's attendance
def show_todays_attendance():
    """Show today's attendance"""
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    filename = f"Attendance/Attendance_{today}.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        st.dataframe(df, use_container_width=True)
        return len(df)
    else:
        st.info("No attendance recorded for today.")
        return 0

# Login Page
def login_page():
    st.title("🎓 Face Recognition Attendance System")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👨‍💼 Admin Login")
        admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
        if st.button("Login as Admin", key="admin_login"):
            if admin_password == get_admin_password():
                st.session_state.logged_in = True
                st.session_state.user_type = "admin"
                st.success("Admin login successful!")
                st.rerun()
            else:
                st.error("Invalid admin password!")
    
    with col2:
        st.subheader("👨‍🎓 Student Login")
        user_id = st.text_input("Student ID", key="student_id")
        if st.button("Login as Student", key="student_login"):
            if user_id:
                is_valid, user_name = verify_user_credentials(user_id)
                if is_valid:
                    st.session_state.logged_in = True
                    st.session_state.user_type = "user"
                    st.session_state.current_user_id = user_id
                    st.session_state.current_user_name = user_name
                    st.success(f"Welcome, {user_name}!")
                    st.rerun()
                else:
                    st.error("Student ID not found in system!")
            else:
                st.error("Please enter your Student ID!")
    
    st.markdown("---")
    st.info("**Admin Panel**: Register new students, train models, manage system")
    st.info("**Student Panel**: Take attendance, view attendance history")

# Admin Panel
def admin_panel():
    st.title("👨‍💼 Admin Panel")
    
    # Logout button in sidebar
    with st.sidebar:
        st.write(f"Logged in as: **Admin**")
        if st.button("🔓 Logout"):
            logout()
    
    # Admin navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Register Student", "🎯 Train Model", "👥 View Students", "📊 Attendance Records", "📧 Email Settings", "🔐 Change Password"])
    
    with tab1:
        st.subheader("Register New Student")
        col1, col2, col3 = st.columns(3)
        with col1:
            student_id = st.text_input("Student ID")
        with col2:
            student_name = st.text_input("Student Name")
        with col3:
            student_email = st.text_input("Student Email (Optional)", placeholder="student@example.com")
        
        if st.button("📸 Capture Images", type="primary"):
            if student_id and student_name:
                if TakeImages(student_id, student_name, student_email):
                    st.info("Student registered successfully! Don't forget to train the model.")
            else:
                st.error("Please enter both Student ID and Name")
    
    with tab2:
        st.subheader("Train Face Recognition Model")
        st.info("Train the model after registering new students to enable face recognition.")
        
        if st.button("🎯 Start Training", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                TrainImages()
    
    with tab3:
        st.subheader("Registered Students")
        student_count = show_all_students()
        st.metric("Total Students", student_count)
    
    with tab4:
        st.subheader("Attendance Records")
        
        # Show today's attendance
        st.write("**Today's Attendance:**")
        today_count = show_todays_attendance()
        st.metric("Today's Attendance", today_count)
        
        # Show all attendance files
        st.write("**All Attendance Records:**")
        attendance_folder = "Attendance/"
        if os.path.exists(attendance_folder):
            files = [f for f in os.listdir(attendance_folder) if f.endswith('.csv')]
            if files:
                selected_file = st.selectbox("Select Date", files)
                if selected_file:
                    df = pd.read_csv(os.path.join(attendance_folder, selected_file))
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=selected_file,
                        mime='text/csv'
                    )
            else:
                st.info("No attendance records found.")
    
    with tab5:
        st.subheader("📧 Email Configuration")
        st.info("Configure email settings to allow students to receive their attendance via email.")
        
        # Get current email config
        current_config = get_email_config()
        
        col1, col2 = st.columns(2)
        with col1:
            sender_email = st.text_input("System Email Address", 
                                       value=current_config.get('sender_email', ''),
                                       placeholder="system@yourdomain.com")
            smtp_server = st.text_input("SMTP Server", 
                                      value=current_config.get('smtp_server', 'smtp.gmail.com'),
                                      help="For Gmail: smtp.gmail.com")
        
        with col2:
            sender_password = st.text_input("Email App Password", 
                                          type="password",
                                          placeholder="Enter app password",
                                          help="For Gmail, use App Password, not your regular password")
            smtp_port = st.text_input("SMTP Port", 
                                    value=current_config.get('smtp_port', '587'),
                                    help="For Gmail: 587")
        
        if st.button("💾 Save Email Configuration", type="primary"):
            if sender_email and sender_password:
                save_email_config(sender_email, sender_password, smtp_server, smtp_port)
                st.success("Email configuration saved successfully!")
                st.info("Students can now send their attendance to their email addresses.")
            else:
                st.error("Please enter both email address and password")
        
        # Email setup instructions
        st.markdown("---")
        st.markdown("### 📋 Email Setup Instructions")
        with st.expander("Gmail Setup Guide"):
            st.markdown("""
            **To use Gmail for sending attendance emails:**
            
            1. **Enable 2-Factor Authentication** on your Gmail account
            2. **Generate App Password**:
               - Go to Google Account settings
               - Security → 2-Step Verification → App passwords
               - Generate password for "Mail"
            3. **Use these settings**:
               - SMTP Server: `smtp.gmail.com`
               - SMTP Port: `587`
               - Email: Your Gmail address
               - Password: The app password (not your Gmail password)
            
            **Security Note:** Use a dedicated email account for the system, not your personal email.
            """)
    
    with tab6:
        st.subheader("Change Admin Password")
        col1, col2, col3 = st.columns(3)
        with col1:
            old_pass = st.text_input("Current Password", type="password")
        with col2:
            new_pass = st.text_input("New Password", type="password")
        with col3:
            confirm_pass = st.text_input("Confirm Password", type="password")
        
        if st.button("🔐 Change Password"):
            save_password(old_pass, new_pass, confirm_pass)

# User Panel
def user_panel():
    st.title(f"👨‍🎓 Student Panel - Welcome {st.session_state.current_user_name}!")
    
    # Logout button in sidebar
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.current_user_name}**")
        st.write(f"Student ID: **{st.session_state.current_user_id}**")
        if st.button("🔓 Logout"):
            logout()
    
    # User navigation
    tab1, tab2, tab3 = st.tabs(["📸 Take Attendance", "📊 My Attendance", "📧 Email Settings"])
    
    with tab1:
        st.subheader("Mark Your Attendance")
        st.info("Click the button below to start the camera and mark your attendance.")
        
        if st.button("📸 Start Attendance Camera", type="primary"):
            TrackImages()
    
    with tab2:
        st.subheader("My Attendance History")
        
        # Get user's attendance
        user_attendance = get_user_attendance(st.session_state.current_user_id)
        
        if user_attendance:
            df_user = pd.DataFrame(user_attendance)
            st.dataframe(df_user, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days Present", len(df_user))
            with col2:
                # Get current month attendance
                current_month = datetime.datetime.now().strftime('%m-%Y')
                monthly_attendance = [record for record in user_attendance 
                                    if current_month in record['Date']]
                st.metric("This Month", len(monthly_attendance))
            with col3:
                # Today's attendance
                today = datetime.datetime.now().strftime('%d-%m-%Y')
                today_attendance = [record for record in user_attendance 
                                  if record['Date'] == today]
                status = "Present" if today_attendance else "Absent"
                st.metric("Today's Status", status)
            
            # Download personal attendance
            csv = df_user.to_csv(index=False)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download My Attendance",
                    data=csv,
                    file_name=f"attendance_{st.session_state.current_user_name}_{st.session_state.current_user_id}.csv",
                    mime='text/csv'
                )
            with col2:
                # Get student email
                student_email = get_student_email(st.session_state.current_user_id)
                if student_email:
                    if st.button("📧 Email My Attendance", type="secondary"):
                        with st.spinner("Sending email..."):
                            if send_attendance_email(student_email, st.session_state.current_user_name, 
                                                   st.session_state.current_user_id, user_attendance):
                                st.success(f"✅ Attendance records sent to {student_email}")
                            else:
                                st.error("❌ Failed to send email. Please contact admin.")
                else:
                    st.info("📧 Set your email address in Email Settings to receive attendance via email")
        else:
            st.info("No attendance records found for your ID.")
    
    with tab3:
        st.subheader("📧 Email Settings")
        st.info("Set your email address to receive attendance records via email.")
        
        # Get current email
        current_email = get_student_email(st.session_state.current_user_id)
        
        # Email input
        new_email = st.text_input("Your Email Address", 
                                 value=current_email if current_email else "",
                                 placeholder="student@example.com",
                                 help="Enter your email address to receive attendance records")
        
        if st.button("💾 Save Email Address", type="primary"):
            if new_email:
                if "@" in new_email and "." in new_email:  # Basic email validation
                    if update_student_email(st.session_state.current_user_id, new_email):
                        st.success("✅ Email address saved successfully!")
                        st.info("You can now receive your attendance records via email from the 'My Attendance' tab.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save email address.")
                else:
                    st.error("❌ Please enter a valid email address.")
            else:
                st.error("❌ Please enter an email address.")
        
        # Test email functionality
        if current_email:
            st.markdown("---")
            st.write(f"**Current Email:** {current_email}")
            
            # Check if email is configured by admin
            email_config = get_email_config()
            if email_config.get('sender_email'):
                if st.button("🧪 Send Test Email", type="secondary"):
                    test_data = [{
                        'ID': st.session_state.current_user_id,
                        'Name': st.session_state.current_user_name,
                        'Date': datetime.datetime.now().strftime('%d-%m-%Y'),
                        'Time': datetime.datetime.now().strftime('%H:%M:%S')
                    }]
                    
                    with st.spinner("Sending test email..."):
                        if send_attendance_email(current_email, st.session_state.current_user_name,
                                               st.session_state.current_user_id, test_data):
                            st.success("✅ Test email sent successfully!")
                        else:
                            st.error("❌ Failed to send test email. Please contact admin.")
            else:
                st.warning("⚠️ Email system not configured by admin. Please contact admin to set up email functionality.")

# Main Streamlit Application
def main():
    # Set page config
    st.set_page_config(
        page_title="Face Recognition Attendance System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Set background image
    set_background_image()
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check login status
    if not st.session_state.logged_in:
        login_page()
    else:
        # Route to appropriate panel
        if st.session_state.user_type == "admin":
            admin_panel()
        elif st.session_state.user_type == "user":
            user_panel()

if __name__ == "__main__":
    main()
