import math
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, flash
import os
from flask_pymongo import PyMongo
from flask_mail import Mail, Message
from oauthlib.oauth2 import WebApplicationClient
from bson.objectid import ObjectId
import json
import numpy as np
from enum import Enum
import warnings
import requests
import threading
import utils
import random
import time
import cv2
try:
    import keyboard
except ImportError:
    keyboard = None
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress general warnings
warnings.filterwarnings("ignore")
# Specific suppression for pkg_resources warning often triggered by libraries
warnings.filterwarnings("ignore", category=UserWarning, module='face_recognition_models')

# Allow OAuth over HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Global variables
studentInfo = None
camera = None
profileName = None

# Flask's Application Configuration
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key')
os.path.dirname("../templates")

# Flask's Database Configuration
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")

# Email Configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])

# Google Auth Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Initialize Client
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Initialize extensions
mail = Mail(app)
mongo = PyMongo(app)

def get_google_provider_cfg():
    """Fetch Google's OpenID configuration."""
    try:
        response = requests.get(GOOGLE_DISCOVERY_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching Google provider config: {e}")
        return {}

def send_warning_mail(violation_type):
    """Send warning email for exam termination."""
    msg = Message('Exam Termination Alert', 
                  sender=app.config['MAIL_DEFAULT_SENDER'], 
                  recipients=['admin@test.com'])
    msg.body = f"An exam was automatically terminated. Reason: {violation_type}. Please check the admin dashboard for evidence."
    try:
        mail.send(msg)
        print(f"Warning email sent for violation: {violation_type}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# Function to show face detection's Rectangle in Face Input Page
def capture_by_frames():
    """Video streaming route for face detection."""
    # We rely on utils.cap being opened by the master_frame_reader thread
    # Removed local fallback to prevent race conditions/conflicts on Windows
    # if not hasattr(utils, 'cap') or utils.cap is None or not utils.cap.isOpened():
    #     print("Camera not open, attempting to open in capture_by_frames")
    #     utils.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Initialize face detector with proper error handling
    try:
        detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        if detector.empty():
            # Fallback to OpenCV built-in path
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"Error loading face detector: {e}")
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Wait for master thread to initialize camera
        if not hasattr(utils, 'cap') or not utils.cap or not utils.cap.isOpened():
            # print("Waiting for camera...")
            time.sleep(0.1)  # Reduced wait time
            continue
            
        frame = utils.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for faster detection
        faces = detector.detectMultiScale(gray, 1.1, 4)  # Optimized parameters
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Reduced thickness for performance
        
        # Encode frame for streaming with optimized quality
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Increased quality
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay for rate limiting the stream
        time.sleep(0.03)  # Reduced delay for smoother streaming

# Function to run Cheat Detection when we start the Application
def start_loop():
    """Start cheat detection threads on first request."""
    # Check if threads are already running to avoid duplicate starts
    if not hasattr(app, 'threads_started'):
        try:
            # Submit tasks to thread pool
            executor.submit(utils.cheat_Detection2)
            executor.submit(utils.cheat_Detection1)
            executor.submit(utils.fr.run_recognition)
            executor.submit(utils.a.record)
            app.threads_started = True
            print("Cheat detection threads started")
        except Exception as e:
            print(f"Error starting cheat detection threads: {e}")

@app.route('/')
@app.route('/main')
def main():
    """Main login page."""
    return render_template('login.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    """Handle exam results."""
    global studentInfo
    if request.method == 'POST':
        result_data = request.form
        # Get suspicion log from utils
        suspicion_data = utils.get_suspicion_log()
        
        # Add to result document
        try:
            # We must update the LAST result saved by this session/process
            # Since write_json increments the ID, the actual result ID is resultId - 1
            current_id = utils.get_resultId() - 1
            mongo.db.results.update_one(
                {"Id": current_id},
                {"$set": {
                    "SuspicionLog": suspicion_data,
                    "EndTime": time.strftime("%H:%M:%S")
                }}
            )
        except Exception as e:
            print(f"Error updating result: {e}")
            
        return render_template("Results.html", result=result_data, studentInfo=studentInfo)
    
    # Handle GET request if needed
    return redirect(url_for('main'))

@app.route('/login', methods=['POST'])
def login():
    """Handle traditional login."""
    global studentInfo
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # MongoDB login logic
        print(f"Attempting login for: {username}")
        
        try:
            user = mongo.db.students.find_one({"Email": username, "Password": password})
            
            if user is None:
                flash('Your Email or Password is incorrect, try again.', 'error')
                return redirect(url_for('main'))
            
            # Set student info
            studentInfo = {
                "Id": str(user['_id']),
                "Name": user['Name'],
                "Email": user['Email'],
                "Password": user['Password']
            }
            
            # Set user in session
            session['user_id'] = str(user['_id'])
            session['user_name'] = user['Name']
            session['user_email'] = user['Email']
            session['user_role'] = user['Role']
            
            if user['Role'] == 'STUDENT':
                utils.Student_Name = user['Name']
                return redirect(url_for('rules'))
            else:
                return redirect(url_for('adminStudents'))
                
        except Exception as e:
            print(f"Login error: {e}")
            flash('An error occurred during login. Please try again.', 'error')
            return redirect(url_for('main'))

@app.route('/logout')
def logout():
    """Logout user and clear session."""
    session.clear()
    utils.Student_Name = None  # Reset student name
    # Stop proctoring if running
    if hasattr(utils, 'stop_proctoring'):
        utils.stop_proctoring()
    return redirect(url_for('main'))

@app.route('/google_login')
def google_login():
    """Initiate Google OAuth login."""
    # Get Google's auth configuration
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        flash('Unable to connect to Google authentication service.', 'error')
        return redirect(url_for('main'))
    
    authorization_endpoint = google_provider_cfg.get("authorization_endpoint")
    if not authorization_endpoint:
        flash('Google authentication endpoint not found.', 'error')
        return redirect(url_for('main'))
    
    # Construct callback URL
    redirect_uri = url_for('google_callback', _external=True)
    
    print(f"DEBUG: Redirect URI generated: {redirect_uri}")
    
    # Prepare request URI
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route('/google_login/callback')
def google_callback():
    """Handle Google OAuth callback."""
    global studentInfo
    
    # Check for error in callback
    error = request.args.get('error')
    if error:
        flash(f'Google authentication error: {error}', 'error')
        return redirect(url_for('main'))
    
    # Get authorization code
    code = request.args.get('code')
    if not code:
        flash('No authorization code received from Google.', 'error')
        return redirect(url_for('main'))
    
    # Get Google provider config
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        flash('Unable to connect to Google authentication service.', 'error')
        return redirect(url_for('main'))
    
    token_endpoint = google_provider_cfg.get('token_endpoint')
    if not token_endpoint:
        flash('Google token endpoint not found.', 'error')
        return redirect(url_for('main'))
    
    # Prepare and send token request
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    
    try:
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
        )
        token_response.raise_for_status()
    except requests.RequestException as e:
        flash(f'Failed to exchange code for token: {e}', 'error')
        return redirect(url_for('main'))
    
    # Parse the tokens
    try:
        client.parse_request_body_response(json.dumps(token_response.json()))
    except Exception as e:
        flash(f'Failed to parse token response: {e}', 'error')
        return redirect(url_for('main'))
    
    # Get user info
    userinfo_endpoint = google_provider_cfg.get('userinfo_endpoint')
    if not userinfo_endpoint:
        flash('Google userinfo endpoint not found.', 'error')
        return redirect(url_for('main'))
    
    uri, headers, body = client.add_token(userinfo_endpoint)
    try:
        userinfo_response = requests.get(uri, headers=headers, data=body)
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
    except requests.RequestException as e:
        flash(f'Failed to get user info from Google: {e}', 'error')
        return redirect(url_for('main'))
    
    # Verify email
    if not userinfo.get('email_verified'):
        flash('Google email not verified.', 'error')
        return redirect(url_for('main'))
    
    # Extract user info
    unique_id = userinfo.get('sub')
    users_email = userinfo.get('email')
    picture = userinfo.get('picture', '')
    users_name = userinfo.get('given_name', users_email.split('@')[0])
    
    # Check if user exists in DB
    try:
        user = mongo.db.students.find_one({"Email": users_email})
        
        if not user:
            # Register new student automatically
            new_student = {
                "Name": users_name,
                "Email": users_email,
                "Password": "GoogleLogin",  # Placeholder
                "Role": "STUDENT"
            }
            mongo.db.students.insert_one(new_student)
            user = mongo.db.students.find_one({"Email": users_email})
        
        # Log them in
        studentInfo = {
            "Id": str(user['_id']),
            "Name": user['Name'],
            "Email": user['Email'],
            "Password": user['Password']
        }
        
        # Set session
        session['user_id'] = str(user['_id'])
        session['user_name'] = user['Name']
        session['user_email'] = user['Email']
        session['user_role'] = user['Role']
        
        if user['Role'] == 'STUDENT':
            utils.Student_Name = user['Name']
            return redirect(url_for('rules'))
        else:
            return redirect(url_for('adminStudents'))
            
    except Exception as e:
        flash(f'Database error during Google login: {e}', 'error')
        return redirect(url_for('main'))

# Student Related Routes
@app.route('/rules')
def rules():
    """Display exam rules."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    return render_template('ExamRules.html')

@app.route('/faceInput')
def faceInput():
    """Face input page for student verification."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    return render_template('ExamFaceInput.html')

@app.route('/video_capture')
def video_capture():
    """Video streaming route for face detection."""
    return Response(capture_by_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/saveFaceInput')
def saveFaceInput():
    """Save face input image."""
    global profileName, studentInfo
    
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    # Session-safe studentInfo retrieval
    if studentInfo is None:
        user = mongo.db.students.find_one({"_id": ObjectId(session['user_id'])})
        if user:
            studentInfo = {
                "Id": str(user['_id']),
                "Name": user['Name'],
                "Email": user['Email'],
                "Password": user['Password']
            }
        else:
            flash('Student information not found.', 'error')
            return redirect(url_for('main'))

    # Capture image from existing webcam if open, otherwise open new
    frame = None
    if hasattr(utils, 'cap') and utils.cap is not None and utils.cap.isOpened():
        success, frame = utils.cap.read()
    else:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cam.isOpened():
            success, frame = cam.read()
            cam.release()
        else:
            success = False

    if not success or frame is None:
        flash('Unable to access camera or capture image. Please try again.', 'error')
        return redirect(url_for('faceInput'))
    
    # Generate profile name (using resultId from utils)
    res_id = utils.get_resultId()
    profileName = f"{studentInfo['Name']}_{res_id:03d}_Profile.jpg"
    
    # Save image locally first
    cv2.imwrite(profileName, frame)
    
    # Move to profiles folder
    try:
        utils.move_file_to_output_folder(profileName, 'Profiles')
    except Exception as e:
        print(f"Error moving profile image: {e}")
        # If move fails, try to just keep it or report error
        flash('Verification image saved with warnings.', 'warning')
    
    return redirect(url_for('confirmFaceInput'))

@app.route('/confirmFaceInput')
def confirmFaceInput():
    """Confirm face input."""
    global profileName
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    # Encode faces for recognition
    try:
        utils.fr.encode_faces()
    except Exception as e:
        print(f"Error encoding faces: {e}")
        flash('Error processing face data. You might need to re-capture.', 'warning')
    
    return render_template('ExamConfirmFaceInput.html', profile=profileName)

@app.route('/systemCheck')
def systemCheck():
    """System check page."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    # Ensure camera is released so browser can access it for system check
    if hasattr(utils, 'cap') and utils.cap is not None:
        utils.cap.release()
        utils.cap = None
        
    return render_template('ExamSystemCheck.html')

@app.route('/systemCheck', methods=["POST"])
def systemCheckRoute():
    """Handle system check AJAX request."""
    if request.method == 'POST':
        examData = request.get_json()
        if not examData:
            return jsonify({"output": "systemCheckError"})
        
        output = 'exam'
        if 'Not available' in examData.get('input', '').split(';'):
            output = 'systemCheckError'
        return jsonify({"output": output})

@app.route('/systemCheckError')
def systemCheckError():
    """System check error page."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    return render_template('ExamSystemCheckError.html')

@app.route('/exam')
def exam():
    """Start exam page."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    # Reset flags
    utils.stop_proctoring_flag = False
    utils.Globalflag = True
    
    # Wait for master_frame_reader to handle camera if needed
    # (It's already running in the background)
    counter = 0
    while (not hasattr(utils, 'cap') or utils.cap is None or not utils.cap.isOpened()) and counter < 20:  # Increased timeout
        time.sleep(0.2)  # Reduced sleep time
        counter += 1
    
    if not hasattr(utils, 'cap') or utils.cap is None or not utils.cap.isOpened():
        flash('Unable to access camera for proctoring.', 'error')
        return redirect(url_for('systemCheck'))
    
    # Reset flags
    utils.stop_proctoring_flag = False
    utils.Globalflag = True
    
    # Reset exam status and violation history
    utils.violation_counts = {}
    utils.exam_status = {'terminated': False, 'violation_type': '', 'evidence_image': ''}
    
    # Start proctoring logic threads
    start_loop()
    
    # Setup keyboard hook
    try:
        keyboard.hook(utils.shortcut_handler)
    except Exception as e:
        print(f"Error setting up keyboard hook: {e}")
    
    return render_template('Exam.html')

@app.route('/exam', methods=["POST"])
def examAction():
    """Handle exam submission."""
    global studentInfo
    link = ''
    if request.method == 'POST':
        examData = request.get_json()
        if not examData:
            return jsonify({"output": "", "link": ""})
            
        # Session-safe studentInfo retrieval
        current_student = studentInfo
        if current_student is None:
            if 'user_id' in session:
                user = mongo.db.students.find_one({"_id": ObjectId(session['user_id'])})
                if user:
                    current_student = {
                        "Id": str(user['_id']),
                        "Name": user['Name'],
                        "Email": user['Email'],
                        "Password": user['Password']
                    }
        
        if current_student is None:
            return jsonify({"output": "Session Expired", "link": "main"})

        resultStatus = ''
        
        if examData.get('input') != '':
            utils.Globalflag = False
            utils.stop_proctoring()
            
            # Handle shortcuts detection
            if hasattr(utils, 'shorcuts') and utils.shorcuts:
                # Individual records are now handled in utils.Shortcut_record_duration
                # We just clear the list here after processing marks
                utils.shorcuts = []
            
            # Calculate scores
            trustScore = utils.get_TrustScore(utils.get_resultId())
            
            try:
                exam_score = float(examData.get('input', 0))
                totalMark = math.floor(exam_score * 6.6667)
            except (ValueError, TypeError):
                totalMark = 0
            
            # Determine status
            if trustScore >= 30:
                status = "Fail(Cheating)"
                link = 'showResultFail'
            else:
                if totalMark < 50:
                    status = "Fail"
                    link = 'showResultFail'
                else:
                    status = "Pass"
                    link = 'showResultPass'
            
            # Extract location and sessionId
            location = examData.get('location', {})
            lat = location.get('latitude', 'Unknown')
            lon = location.get('longitude', 'Unknown')
            sessionId = examData.get('sessionId', 'Unknown')

            # Save result to JSON
            result_entry = {
                "Id": utils.get_resultId(),
                "Name": current_student['Name'],
                "TotalMark": totalMark,
                "TrustScore": max(100 - trustScore, 0),
                "Status": status,
                "Date": time.strftime("%Y-%m-%d", time.localtime(time.time())),
                "StId": current_student['Id'],
                "Link": profileName if profileName else 'avatar.svg',
                "StartTime": time.strftime("%H:%M:%S"),
                "Location": f"{lat}, {lon}",
                "SessionId": sessionId
            }
            utils.write_json(result_entry, "result.json")

            # Save result to MongoDB
            try:
                mongo.db.results.update_one(
                    {"Id": result_entry["Id"]},
                    {"$set": result_entry},
                    upsert=True
                )
            except Exception as e:
                print(f"Error saving result to MongoDB: {e}")
            
            resultStatus = f"{current_student['Name']};{totalMark};{status};{time.strftime('%Y-%m-%d', time.localtime(time.time()))}"
        else:
            utils.Globalflag = True
            print('Exam started or empty submission')
        
        return jsonify({"output": resultStatus, "link": link})

@app.route('/upload_recording', methods=['POST'])
def upload_recording():
    """Handle screen recording uploads."""
    if 'recording' not in request.files:
        return jsonify({"error": "No recording file"}), 400
    
    file = request.files['recording']
    sessionId = request.form.get('sessionId', str(int(time.time())))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure recordings directory exists
    recording_dir = os.path.join('static', 'recordings')
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)

    # Save file with sessionId
    filename = f"{sessionId}.webm"
    file.save(os.path.join(recording_dir, filename))
    
    return jsonify({"success": True, "filename": filename})

@app.route('/upload_audio_violation', methods=['POST'])
def upload_audio_violation():
    """Handle audio evidence uploads for noise violations."""
    if 'audio_evidence' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    file = request.files['audio_evidence']
    sessionId = request.form.get('sessionId', str(int(time.time())))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure violations directory exists
    violation_dir = os.path.join('static', 'Violations')
    if not os.path.exists(violation_dir):
        os.makedirs(violation_dir)

    # Save file
    timestamp = int(time.time())
    filename = f"noise_violation_{sessionId}_{timestamp}.webm"
    file.save(os.path.join(violation_dir, filename))
    
    # Log the violation here to ensure we have the link
    # We can either update the last violation or creating a new one. 
    # Since the frontend calls reportViolation separately, we might just want to return the filename
    # and let the frontend/backend correlation handle it, OR we just log it here as a "Evidence"
    
    # Let's log it as a specific evidence entry or update the violation log?
    # Simplest: Update the noise violation logic to include this file.
    # For now, just save it and return success, assuming the admin can find it by timestamp/session
    
    return jsonify({"success": True, "filename": filename})

@app.route('/api/violation', methods=['POST'])
def report_violation():
    """Handle violations reported from frontend (e.g. Fullscreen Exit)."""
    data = request.json
    violation_type = data.get('type', 'Unknown Violation')
    details = data.get('details', '')
    
    print(f"Violation reported from frontend: {violation_type} - {details}")
    
    # Increment violation count
    utils.violation_counts[violation_type] = utils.violation_counts.get(violation_type, 0) + 1
    count = utils.violation_counts[violation_type]
    
    # Capture evidence frame if possible
    img_filename = None
    if utils.cap is not None and utils.cap.isOpened():
        ret, frame = utils.cap.read()
        if ret:
            img_filename = f"api_violation_{int(time.time())}.jpg"
            cv2.imwrite(img_filename, frame)
            utils.move_file_to_output_folder(img_filename, 'Violations')

    # Log to violation.json
    violation_entry = {
        "Name": violation_type,
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": "N/A",
        "Mark": 10, # Arbitrary mark for now
        "Link": f"{details} (Count: {count})",
        "Image": img_filename, # Added Image field
        "RId": utils.get_resultId()
    }
    utils.write_json(violation_entry, 'violation.json')
    
    # Graduated enforcement logic
    if violation_type == "Fullscreen Exit":
        if count >= 2:
            utils.terminate_exam(violation_type)
            return jsonify({"success": True, "action": "terminated"})
        else:
            return jsonify({"success": True, "action": "warning", "message": "First warning: Fullscreen is mandatory. Next exit will terminate your exam."})
    
    # Critical violations (like phone detection) trigger immediate termination
    if violation_type in ["Mobile Phone Detected", "Electronic Device"]:
        utils.terminate_exam(violation_type)
        return jsonify({"success": True, "action": "terminated"})
        
    return jsonify({"success": True, "action": "logged"})

@app.route('/api/check_person_status', methods=['GET'])
def check_person_status():
    """Check if person is detected in camera."""
    status = getattr(utils, 'no_person_status', {'detected': True, 'start_time': None})
    # print(f"DEBUG: Status Check - Detected: {status['detected']}, Start: {status['start_time']}") 
    
    if not status['detected'] and status['start_time'] is not None:
        elapsed = time.time() - status['start_time']
        
        # Immediate Countdown Phase
        if elapsed < 30:
            remaining = 30 - elapsed
            return jsonify({
                "status": "countdown_phase", 
                "message": "Return to camera view within:",
                "elapsed": elapsed,
                "remaining": int(remaining),
                "total_countdown": 30,
                "warning": True
            })
        else:
            # Countdown finished - terminate the exam
            if not utils.exam_status.get('terminated', False):
                # Capture evidence frame before termination
                evidence_frame = None
                if utils.cap is not None and utils.cap.isOpened():
                    ret, frame = utils.cap.read()
                    if ret:
                        evidence_frame = frame
                utils.terminate_exam("No participant detected in camera feed - Countdown expired", evidence_frame)
            return jsonify({"status": "terminated", "warning": True})
            
    return jsonify({"status": "ok", "warning": False})

@app.route('/api/reset_absence', methods=['POST'])
def reset_absence():
    """Manually reset the absence timer (called by I'm Back button)."""
    if hasattr(utils, 'no_person_status'):
        utils.no_person_status['detected'] = True
        utils.no_person_status['start_time'] = None
        print("✅ Absence timer manually reset via API")
    return jsonify({"success": True})

@app.route('/showResultPass/<result_status>')
def showResultPass(result_status):
    """Show pass result page."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    return render_template('ExamResultPass.html', result_status=result_status)

@app.route('/showResultFail/<result_status>')
def showResultFail(result_status):
    """Show fail result page."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        flash('Please login as a student first.', 'error')
        return redirect(url_for('main'))
    
    return render_template('ExamResultFail.html', result_status=result_status)

# AI Chat Bot Route
@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot messages."""
    if 'user_id' not in session or session.get('user_role') != 'STUDENT':
        return jsonify({"response": "Please login first."})
    
    data = request.get_json()
    user_message = data.get('message', '').strip().lower()
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})
    
    # Simple Rule-based Logic (Mock AI)
    response = "I am not sure about that. Please focus on the exam questions."
    
    if any(word in user_message for word in ["time", "duration", "remaining"]):
        response = "The timer at the top right shows the remaining time. Please check there."
    elif any(word in user_message for word in ["rule", "cheat", "violation", "allowed"]):
        response = "Strict rules are in place. Creating noise, looking away, or using other devices will lead to termination."
    elif any(word in user_message for word in ["submit", "finish", "end", "complete"]):
        response = "You can submit the exam once you have answered all questions. The submit button will appear when you complete all questions."
    elif any(word in user_message for word in ["hello", "hi", "hey", "greetings"]):
        response = "Hello! I am here to assist you with any technical or rule-related queries."
    elif any(word in user_message for word in ["question", "answer", "solution", "help with"]):
        response = "I cannot help you with the answers to the exam questions. Please rely on your own knowledge. Good luck!"
    elif any(word in user_message for word in ["camera", "mic", "audio", "video"]):
        response = "Camera and microphone are being monitored for proctoring purposes. Ensure they are working properly."
    elif any(word in user_message for word in ["technical", "problem", "issue", "error"]):
        response = "If you're experiencing technical issues, please try refreshing the page. If the problem persists, contact the administrator."
    
    # Save Chat Log to MongoDB
    chat_entry = {
        "sender": "User",
        "message": user_message,
        "time": time.strftime("%H:%M:%S")
    }
    bot_entry = {
        "sender": "Bot",
        "message": response,
        "time": time.strftime("%H:%M:%S")
    }
    
    # Update the result document with chat history
    try:
        result_id = utils.get_resultId()
        mongo.db.results.update_one(
            {"Id": result_id},
            {"$push": {"ChatHistory": {"$each": [chat_entry, bot_entry]}}},
            upsert=True  # Create document if it doesn't exist
        )
    except Exception as e:
        print(f"Error saving chat log: {e}")
        
    return jsonify({"response": response})

@app.route('/api/lockdown_heartbeat', methods=['POST'])
def lockdown_heartbeat():
    """Receive heartbeat from Chrome Extension."""
    data = request.json
    utils.extension_heartbeat_time = time.time()
    # print(f"Heartbeat received: {data}")
    return jsonify({"success": True})

# Admin Related Routes
@app.route('/adminResults')
def adminResults():
    """Admin results page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    # Fetch results from MongoDB, fallback to JSON if empty
    try:
        results = list(mongo.db.results.find().sort("Id", -1))
        if not results:
            results = utils.getResults()
    except Exception as e:
        print(f"Error fetching results from MongoDB: {e}")
        results = utils.getResults()
        
    return render_template('Results.html', results=results)

@app.route('/adminResultDetails/<int:id>')
def adminResultDetails(id):
    """Admin result details page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    # Fetch result details from MongoDB, fallback to JSON
    try:
        mongo_result = mongo.db.results.find_one({"Id": id})
        resultDetials = utils.getResultDetails(id)
        if mongo_result:
            # Fill missing required keys from JSON fallback if they are missing in MongoDB
            for key in ["TotalMark", "TrustScore", "Status", "Name", "Date"]:
                if key not in mongo_result and resultDetials['Result'] and key in resultDetials['Result'][0]:
                    mongo_result[key] = resultDetials['Result'][0][key]
            
            # Prefer MongoDB metadata (includes Location, SessionId)
            resultDetials['Result'] = [mongo_result]
    except Exception as e:
        print(f"Error fetching result details from MongoDB: {e}")
        resultDetials = utils.getResultDetails(id)
        
    return render_template('ResultDetails.html', resultDetials=resultDetials)

@app.route('/enable_retest/<int:id>')
def enable_retest(id):
    """Enable retest for a student."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    try:
        # Delete the result to enable retest
        result = mongo.db.results.delete_one({"Id": id})
        if result.deleted_count > 0:
            flash(f"Retest enabled for exam ID {id}. The student can now take the exam again.", "success")
        else:
            flash(f"No exam found with ID {id}.", "warning")
    except Exception as e:
        flash(f"Error enabling retest: {str(e)}", "error")
    
    return redirect(url_for('adminResults'))

@app.route('/live_monitoring')
def live_monitoring():
    """Live monitoring page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    return render_template('LiveMonitoring.html')

@app.route('/adminResultDetailsVideo/<videoInfo>')
def adminResultDetailsVideo(videoInfo):
    """Admin result details video page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    return render_template('ResultDetailsVideo.html', videoInfo=videoInfo)

@app.route('/adminFullRecording/<sessionId>')
def adminFullRecording(sessionId):
    """Admin full session recording page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    return render_template('FullRecording.html', sessionId=sessionId)

@app.route('/adminProfile')
def adminProfile():
    """Admin profile page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    global studentInfo
    return render_template('AdminProfile.html', admin=studentInfo)

@app.route('/adminStudents')
def adminStudents():
    """Admin students management page."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    try:
        students = list(mongo.db.students.find({"Role": "STUDENT"}))
        # Convert ObjectId to string for template
        for student in students:
            student['_id'] = str(student['_id'])
    except Exception as e:
        print(f"Error fetching students: {e}")
        students = []
        flash('Error loading students data.', 'error')
    
    return render_template('Students.html', students=students)

@app.route('/insertStudent', methods=['POST'])
def insertStudent():
    """Insert new student."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    if request.method == "POST":
        name = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not all([name, email, password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('adminStudents'))
        
        try:
            # Check if email already exists
            existing = mongo.db.students.find_one({"Email": email})
            if existing:
                flash('Email already exists.', 'error')
                return redirect(url_for('adminStudents'))
            
            # Insert new student
            mongo.db.students.insert_one({
                "Name": name,
                "Email": email,
                "Password": password,
                "Role": "STUDENT"
            })
            flash('Student added successfully.', 'success')
        except Exception as e:
            flash(f'Error adding student: {str(e)}', 'error')
        
        return redirect(url_for('adminStudents'))

@app.route('/exam_terminated')
def exam_terminated():
    """Exam terminated page."""
    violation_type = utils.exam_status.get('violation_type', 'Unknown Violation')
    evidence_image = utils.exam_status.get('evidence_image', '')
    return render_template('ExamTerminated.html', violation_type=violation_type, evidence_image=evidence_image)

@app.route('/check_exam_status', methods=['GET'])
def check_exam_status():
    """Check exam status for AJAX polling."""
    status = 'terminated' if utils.exam_status.get('terminated', False) else 'running'
    return jsonify({'status': status})

@app.route('/deleteStudent/<string:stdId>', methods=['GET'])
def deleteStudent(stdId):
    """Delete a student."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    try:
        result = mongo.db.students.delete_one({"_id": ObjectId(stdId)})
        if result.deleted_count > 0:
            flash('Student deleted successfully.', 'success')
        else:
            flash('Student not found.', 'warning')
    except Exception as e:
        flash(f'Error deleting student: {str(e)}', 'error')
    
    return redirect(url_for('adminStudents'))

@app.route('/updateStudent', methods=['POST', 'GET'])
def updateStudent():
    """Update student information."""
    if 'user_id' not in session or session.get('user_role') != 'ADMIN':
        flash('Please login as an admin first.', 'error')
        return redirect(url_for('main'))
    
    if request.method == 'POST':
        id_data = request.form.get('id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        if not all([id_data, name, email, password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('adminStudents'))
        
        try:
            result = mongo.db.students.update_one(
                {"_id": ObjectId(id_data)},
                {"$set": {"Name": name, "Email": email, "Password": password}}
            )
            if result.modified_count > 0:
                flash('Student updated successfully.', 'success')
            else:
                flash('No changes made or student not found.', 'warning')
        except Exception as e:
            flash(f'Error updating student: {str(e)}', 'error')
        
        return redirect(url_for('adminStudents'))

def initialize_database():
    """Initialize database with default users if needed."""
    try:
        with app.app_context():
            # Check if dummy student exists
            if not mongo.db.students.find_one({"Email": "student@test.com"}):
                mongo.db.students.insert_one({
                    "Name": "Dummy Student",
                    "Email": "student@test.com",
                    "Password": "password",
                    "Role": "STUDENT"
                })
                print("✓ Dummy student created: student@test.com / password")
            
            # Check if dummy admin exists
            if not mongo.db.students.find_one({"Email": "admin@test.com"}):
                mongo.db.students.insert_one({
                    "Name": "Dummy Admin",
                    "Email": "admin@test.com",
                    "Password": "password",
                    "Role": "ADMIN"
                })
                print("✓ Dummy admin created: admin@test.com / password")
            
            # Create indexes for better performance
            mongo.db.students.create_index([("Email", 1)], unique=True)
            mongo.db.results.create_index([("Id", 1)])
            mongo.db.results.create_index([("StId", 1)])
            print("✓ Database indexes created")
            
    except Exception as e:
        print(f"✗ Error initializing database: {e}")

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize database
    initialize_database()
    
    # Print startup information
    print("\n" + "="*50)
    print("Online Exam Proctor System")
    print("="*50)
    print(f"Application URL: http://127.0.0.1:5000")
    
    print("="*50 + "\n")
    
    # Run the application
    # CRITICAL: use_reloader=False prevents double execution of global code (camera init)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)