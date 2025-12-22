import sys
import face_recognition
from concurrent.futures import ThreadPoolExecutor
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
import os
import json
import shutil
import threading
from multiprocessing import Process
import datetime
import subprocess

# Defensive imports for GUI/Audio (may be missing in cloud environments)
try:
    import keyboard
except ImportError:
    keyboard = None
    print("Warning: keyboard module not found. Desktop hooks disabled.")

try:
    import pyautogui
except ImportError:
    pyautogui = None
    print("Warning: pyautogui module not found. Desktop automation disabled.")

try:
    import pygetwindow as gw
except ImportError:
    gw = None
    print("Warning: pygetwindow module not found. Window tracking disabled.")

try:
    import pyperclip
except ImportError:
    pyperclip = None
    print("Warning: pyperclip module not found.")

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("Warning: pyaudio module not found. Audio recording disabled.")

import webbrowser
from ultralytics import YOLO
import struct
import wave

#Variables
#All Related
Globalflag = False
stop_proctoring_flag = False
Student_Name = ''
shortcut_flag = False
shortcut_event_name = ""
exam_status = {'terminated': False, 'violation_type': '', 'evidence_image': ''}
violation_counts = {} # Tracks counts of specific violations per session
no_person_status = {'detected': True, 'start_time': None}
extension_heartbeat_time = 0


# Result ID Initialization
def fetch_last_id():
    try:
        if os.path.exists('result.json'):
            with open('result.json', 'r') as f:
                data = json.load(f)
                if data:
                    return max(item.get('Id', 0) for item in data)
    except Exception as e:
        print(f"Error loading result ID: {e}")
    return 0

resultId = fetch_last_id() + 1

def stop_proctoring():
    global stop_proctoring_flag, Globalflag, cap
    print("Stopping proctoring...")
    stop_proctoring_flag = True
    Globalflag = False
    if cap is not None and cap.isOpened():
        cap.release()

def terminate_exam(violation_type, frame=None):
    global stop_proctoring_flag, Globalflag, exam_status, cap
    print(f"TERMINATING EXAM: {violation_type}")
    
    # Capture evidence frame if not provided
    if frame is None and cap is not None and cap.isOpened():
        ret, captured_frame = cap.read()
        if ret:
            frame = captured_frame
            
    stop_proctoring_flag = True
    Globalflag = False
    exam_status['terminated'] = True
    exam_status['violation_type'] = violation_type
    
    if frame is not None:
        # Save termination evidence image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"termination_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        try:
            move_file_to_output_folder(filename, 'Violations')
            exam_status['evidence_image'] = filename
            print(f"Evidence saved: {filename}")
        except Exception as e:
            print(f"Error saving evidence: {e}")
    
    if cap is not None:
        cap.release()
        cap = None # Ensure it's cleared

start_time = [0, 0, 0, 0, 0]
end_time = [0, 0, 0, 0, 0]
recorded_durations = []
suspicion_log = [] # Time-series log for cheat probability
prev_state = ['Verified Student appeared', "Forward", "Only one person is detected", "Stay in the Test", "No Electronic Device Detected"]
flag = [False, False, False, False, False]
# Start a background timer to log suspicion score
def log_suspicion():
    while not stop_proctoring_flag:
        try:
            # Calculate suspicion score based on active flags
            # Each True flag adds 0.2 to the score (max 1.0)
            score = sum([1 for f in flag if f]) * 0.2
            
            # Artificial "jitter" or cumulative increase if violations persist
            # (To match the "map" requested which looks like a probability curve)
            if score > 0 and len(suspicion_log) > 0:
                 # If violation continues, probability increases slightly
                 score = min(1.0, score + (suspicion_log[-1]['score'] * 0.1))
            
            # Reset slightly if no violation (decay) to make it dynamic
            if score == 0 and len(suspicion_log) > 0 and suspicion_log[-1]['score'] > 0:
                 score = max(0.0, suspicion_log[-1]['score'] - 0.05)

            timestamp = time.strftime("%H:%M:%S")
            suspicion_log.append({"time": timestamp, "score": round(score, 2)})
            
            # Keep log size manageable (e.g., last 1 hour)
            if len(suspicion_log) > 3600:
                suspicion_log.pop(0)
                
            time.sleep(1) # Log every second
        except Exception:
             pass

# Start the logging thread
log_thread = threading.Thread(target=log_suspicion, daemon=True)
log_thread.start()

# Default dimensions for writers (will be updated when camera opens)
width, height = 640, 480
EDWidth, EDHeight = 640, 480

video = [(str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4")]
writer = [cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[2], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[3], cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080)), cv2.VideoWriter(video[4], cv2.VideoWriter_fourcc(*'mp4v'), 20 , (EDWidth,EDHeight))]
#More than One Person Related
# MediaPipe face detection - using OpenCV as fallback for Python 3.12 compatibility
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mpFaceDetection = mp.solutions.face_detection
        mpDraw = mp.solutions.drawing_utils
        faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.75)
    else:
        # MediaPipe 0.10+ doesn't have solutions API - use OpenCV fallback
        mpFaceDetection = None
        mpDraw = None
        faceDetection = None
        print("Warning: Using OpenCV for face detection (MediaPipe solutions not available)")
except Exception as e:
    print(f"MediaPipe initialization error: {e}")
    mpFaceDetection = None
    mpDraw = None
    faceDetection = None
#Screen Related
shorcuts = []
active_window = None # Store the initial active window and its title
active_window_title = "Exam — Mozilla Firefox"
exam_window_title = active_window_title
#ED Related
my_file = open("utils/coco.txt", "r") # opening the file in read mode
data = my_file.read() # reading the file
class_list = data.split("\n") # replacing end splitting the text | when newline ('\n') is seen.
my_file.close()
detected_things = []
detection_colors = [] # Generate random colors for class list
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))
model = YOLO("yolov8n.pt", "v8") # load a pretrained YOLOv8n model
EDFlag = False
#Voice Related
TRIGGER_RMS = 10  # start recording above 10
RATE = 16000  # sample rate
TIMEOUT_SECS = 3  # silence time after which recording stops
FRAME_SECS = 0.25  # length of frame(chunks) to be processed at once in secs
CUSHION_SECS = 1  # amount of recording before and after sound
SHORT_NORMALIZE = (1.0 / 32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SHORT_WIDTH = 2
CHUNK = int(RATE * FRAME_SECS)
CUSHION_FRAMES = int(CUSHION_SECS / FRAME_SECS)
TIMEOUT_FRAMES = int(TIMEOUT_SECS / FRAME_SECS)
f_name_directory = os.path.join(os.getcwd(), 'static', 'OutputAudios')
if not os.path.exists(f_name_directory):
    os.makedirs(f_name_directory)
# Capture
cap = None
global_frame = None
frame_lock = threading.Lock()

def master_frame_reader():
    """Centralized thread to read frames from camera and share with other threads."""
    global global_frame, cap, Globalflag, stop_proctoring_flag
    print("Master Frame Reader thread started")
    while not stop_proctoring_flag:
        # Auto-initialize if missing
        # CRITICAL: Check flag AGAIN before opening to prevent race condition where stop_proctoring closes it 
        # and we immediately re-open it before exiting loop.
        if stop_proctoring_flag: 
            break

        if cap is None or not cap.isOpened():
            try:
                print("Master Reader: Opening Camera...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("Master Reader: Failed to open camera, retrying...")
                    time.sleep(1)
                    continue
                else:
                    print("Master Reader: Camera opened successfully.")
            except Exception as e:
                print(f"Master Reader: Error opening camera: {e}")
                time.sleep(1)
                continue

        if cap is not None and cap.isOpened():
            try:
                success, frame = cap.read()
                if success:
                    with frame_lock:
                        global_frame = frame.copy() if frame is not None else None
                else:
                    print("Master Frame Reader: Failed to read frame from camera")
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in master_frame_reader: {e}")
                time.sleep(0.01)
        else:
            # Idle for a bit if camera not active
            time.sleep(0.1)
    print("Master Frame Reader thread stopped")

def get_frame():
    """Retrieve the latest frame from the shared buffer."""
    global global_frame
    with frame_lock:
        return global_frame.copy() if global_frame is not None else None


#Database and Files Related
# function to add data to JSON
def write_json(new_data, filename='violation.json'):
    global resultId
    file_data = []
    
    # Ensure file exists and is initialized
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
    
    try:
        with open(filename, 'r+') as file:
            try:
                # First we load existing data into a list
                file_data = json.load(file)
            except json.JSONDecodeError:
                # If file is empty or malformed
                file_data = []
            
            # Join new_data with file_data
            file_data.append(new_data)
            
            # Sets file's current position at offset 0
            file.seek(0)
            file.truncate() # Clear existing content
            
            # convert back to json
            json.dump(file_data, file, indent=4)
    except Exception as e:
        print(f"Error writing to {filename}: {e}")
    
    # Increment resultId if we just saved a result
    if filename == 'result.json':
        resultId += 1

#Function to move the files to the Output Folders
def move_file_to_output_folder(file_name, folder_name='OutputVideos'):
    # Get the current working directory (project folder)
    current_directory = os.getcwd()
    # Define the destination directory
    destination_dir = os.path.join(current_directory, 'static', folder_name)
    
    # Ensure the destination folder exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    # Define the paths for the source file and destination folder
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(destination_dir, file_name)
    
    try:
        # Use 'shutil.move' to move the file to the destination folder
        shutil.move(source_path, destination_path)
        print(f"File '{file_name}' moved to 'static/{folder_name}'")
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in the project folder.")
    except shutil.Error as e:
        print(f"Error: Failed to move the file. {e}")

#Function to reduce video file's data rate to 100 kbps
def reduceBitRate (input_file,output_file):
   target_bitrate = "1000k"  # Set your desired target bitrate here
   # Specify the full path to the FFmpeg executable
   ffmpeg_path = "ffmpeg"  # Assumes ffmpeg is in PATH, or provide relative path
   # Run FFmpeg command to lower the bitrate
   command = [
      ffmpeg_path,
      "-i", input_file,
      "-b:v", target_bitrate,
      "-c:v", "libx264",
      "-c:a", "aac",
      "-strict", "experimental",
      "-b:a", "192k",
      output_file
   ]
   subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   print("Bitrate conversion completed.")

#Recordings related
#Recording Function for Face Verification
def faceDetectionRecording(img, text):
    global start_time, end_time, recorded_durations, prev_state, flag, writer, width, height
    print("Running FaceDetection Recording Function")
    print(text)
    if text != 'Verified Student appeared' and prev_state[0] == 'Verified Student appeared':
        start_time[0] = time.time()
        for _ in range(2):
            writer[0].write(img)
    elif text != 'Verified Student appeared' and str(text) == prev_state[0] and (time.time() - start_time[0]) > 30:
        flag[0] = True
        for _ in range(2):
            writer[0].write(img)
        
        # Immediate Termination on confirmed violation (30s)
        if not stop_proctoring_flag: # Ensure we haven't already terminated
            writer[0].release()
            end_time[0] = time.time()
            duration = math.ceil((end_time[0] - start_time[0]) / 3)
            outputVideo = 'FDViolation' + video[0]
            # Save violation evidence image
            img_filename = f"violation_{int(time.time())}.jpg"
            cv2.imwrite(img_filename, img)
            move_file_to_output_folder(img_filename, 'Violations')
            
            FDViolation = {
                "Name": prev_state[0],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[0])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(2 * duration),
                "Link": outputVideo,
                "Image": img_filename, # Added Image field
                "RId": get_resultId()
            }
            recorded_durations.append(FDViolation)
            write_json(FDViolation)
            reduceBitRate(video[0], outputVideo)
            move_file_to_output_folder(outputVideo)
            terminate_exam(FDViolation["Name"], img)
            
            # Reset/Cleanup
            os.remove(video[0])
            video[0] = str(random.randint(1, 50000)) + ".mp4"
            writer[0] = cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
            flag[0] = False

    elif text != 'Verified Student appeared' and str(text) == prev_state[0] and (time.time() - start_time[0]) <= 30:
        flag[0] = False
        for _ in range(2):
            writer[0].write(img)
            
    else:
        # Violation ended or changed state (but if we already terminated, this might not matter much)
        if prev_state[0] != "Verified Student appeared" and not stop_proctoring_flag:
            # Only save if we haven't terminated yet (i.e. short violation that didn't trigger terminate)
            writer[0].release()
            end_time[0] = time.time()
            # If it was flagged (shouldn't happen here if we terminate on flag, but for safety)
            video[0] = str(random.randint(1, 50000)) + ".mp4"
            writer[0] = cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
            flag[0] = False
    prev_state[0] = text

#Recording Function for Head Movement Detection
def Head_record_duration(text,img):
    global start_time, end_time, recorded_durations, prev_state, flag,writer, width, height
    print("Running HeadMovement Recording Function")
    print(text)
    if text != "Forward":
        if str(text) != prev_state[1] and prev_state[1] == "Forward":
            start_time[1] = time.time()
            for _ in range(2):
                writer[1].write(img)
        elif str(text) != prev_state[1] and prev_state[1] != "Forward":
            writer[1].release()
            end_time[1] = time.time()
            duration = math.ceil((end_time[1] - start_time[1])/7)
            outputVideo = 'HeadViolation' + video[1]
            HeadViolation = {
                "Name": prev_state[1],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[1])),
                "Duration": str(duration) + " seconds",
                "Mark": duration,
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[1]:
                recorded_durations.append(HeadViolation)
                write_json(HeadViolation)
                reduceBitRate(video[1], outputVideo)
                move_file_to_output_folder(outputVideo)
                terminate_exam(HeadViolation["Name"], img)
            os.remove(video[1])
            print(recorded_durations)
            start_time[1] = time.time()
            video[1] = str(random.randint(1, 50000)) + ".mp4"
            writer[1] = cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[1] = False
        elif str(text) == prev_state[1] and (time.time() - start_time[1]) > 3:
            flag[1] = True
            for _ in range(2):
                writer[1].write(img)
            
            # Immediate Termination
            if not stop_proctoring_flag:
                writer[1].release()
                end_time[1] = time.time()
                duration = math.ceil((end_time[1] - start_time[1])/7)
                outputVideo = 'HeadViolation' + video[1]
                # Save violation evidence image
                img_filename = f"head_violation_{int(time.time())}.jpg"
                cv2.imwrite(img_filename, img)
                move_file_to_output_folder(img_filename, 'Violations')
                
                HeadViolation = {
                    "Name": prev_state[1],
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[1])),
                    "Duration": str(duration) + " seconds",
                    "Mark": duration,
                    "Link": outputVideo,
                    "Image": img_filename, # Added Image field
                    "RId": get_resultId()
                }
                recorded_durations.append(HeadViolation)
                write_json(HeadViolation)
                reduceBitRate(video[1], outputVideo)
                move_file_to_output_folder(outputVideo)
                terminate_exam(HeadViolation["Name"], img)
                
                os.remove(video[1])
                start_time[1] = time.time()
                video[1] = str(random.randint(1, 50000)) + ".mp4"
                writer[1] = cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
                flag[1] = False

        elif str(text) == prev_state[1] and (time.time() - start_time[1]) <= 3:
            flag[1] = False
            for _ in range(2):
                writer[1].write(img)
        prev_state[1] = text
    else:
        if prev_state[1] != "Forward" and not stop_proctoring_flag:
            writer[1].release()
            end_time[1] = time.time()
            os.remove(video[1])
            video[1] = str(random.randint(1, 50000)) + ".mp4"
            writer[1] = cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[1] = False
        prev_state[1] = text

#Recording Function for More than one person Detection
def MTOP_record_duration(text, img):
    global start_time, end_time, recorded_durations, prev_state, flag, writer, width, height
    print("Running MTOP Recording Function")
    print(text)
    if text != 'Only one person is detected' and prev_state[2] == 'Only one person is detected':
        start_time[2] = time.time()
        for _ in range(2):
            writer[2].write(img)
    elif text != 'Only one person is detected' and str(text) == prev_state[2] and (time.time() - start_time[2]) > 3:
        flag[2] = True
        for _ in range(2):
            writer[2].write(img)
            
        # Immediate Termination
        if not stop_proctoring_flag:
            writer[2].release()
            end_time[2] = time.time()
            duration = math.ceil((end_time[2] - start_time[2])/3)
            outputVideo = 'MTOPViolation' + video[2]
            # Save violation evidence image
            img_filename = f"mtop_violation_{int(time.time())}.jpg"
            cv2.imwrite(img_filename, img)
            move_file_to_output_folder(img_filename, 'Violations')
            
            MTOPViolation = {
                "Name": prev_state[2],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[2])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(1.5 * duration),
                "Link": outputVideo,
                "Image": img_filename, # Added Image field
                "RId": get_resultId()
            }
            recorded_durations.append(MTOPViolation)
            write_json(MTOPViolation)
            reduceBitRate(video[2], outputVideo)
            move_file_to_output_folder(outputVideo)
            terminate_exam(MTOPViolation["Name"], img)
            
            os.remove(video[2])
            video[2] = str(random.randint(1, 50000)) + ".mp4"
            writer[2] = cv2.VideoWriter(video[2], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[2] = False

    elif text != 'Only one person is detected' and str(text) == prev_state[2] and (time.time() - start_time[2]) <= 3:
        flag[2] = False
        for _ in range(2):
            writer[2].write(img)
    else:
        if prev_state[2] != "Only one person is detected" and not stop_proctoring_flag:
            writer[2].release()
            end_time[2] = time.time()
            os.remove(video[2])
            video[2] = str(random.randint(1, 50000)) + ".mp4"
            writer[2] = cv2.VideoWriter(video[2], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[2] = False
    prev_state[2] = text

def Shortcut_record_duration(text, img):
    global start_time, end_time, flag, writer, width, height, video
    print(f"Recording Shortcut Evidence: {text}")
    
    # We record for at least 3 seconds (approx 60 frames at 20fps)
    # Since this is triggered once, we'll start a simple recording
    if not flag[0]:
        start_time[0] = time.time()
        flag[0] = True
        
    writer[0].write(img)
    
    # If 3 seconds passed
    if (time.time() - start_time[0]) > 3:
        writer[0].release()
        outputVideo = 'ShortcutViolation' + video[0]
        # Save violation evidence image
        img_filename = f"shortcut_violation_{int(time.time())}.jpg"
        cv2.imwrite(img_filename, img)
        move_file_to_output_folder(img_filename, 'Violations')
        
        shortcutViolation = {
            "Name": text,
            "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[0])),
            "Duration": "3 seconds",
            "Mark": 5,
            "Link": outputVideo,
            "Image": img_filename, # Added Image field
            "RId": get_resultId()
        }
        write_json(shortcutViolation)
        reduceBitRate(video[0], outputVideo)
        move_file_to_output_folder(outputVideo)
        
        # Reset writer for next use
        os.remove(video[0])
        video[0] = str(random.randint(1, 50000)) + ".mp4"
        writer[0] = cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        flag[0] = False

#Recording Function for Screen Detection
def SD_record_duration(text, img):
    global start_time, end_time, prev_state, flag, writer, width, height
    print("Running SD Recording Function")
    print(text)
    if text != "Stay in the Test" and prev_state[3] == "Stay in the Test":
        start_time[3] = time.time()
        print(f"Start SD Recording, start time is {start_time[3]} and array is {start_time}")
        for _ in range(2):
            writer[3].write(img)
    elif text != "Stay in the Test" and str(text) == prev_state[3] and (time.time() - start_time[3]) > 3:
        flag[3] = True
        for _ in range(2):
            writer[3].write(img)
            
        # Immediate Termination
        if not stop_proctoring_flag:
            writer[3].release()
            end_time[3] = time.time()
            duration = math.ceil((end_time[3] - start_time[3])/4)
            outputVideo = 'SDViolation' + video[3]
            SDViolation = {
                "Name": prev_state[3],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[3])),
                "Duration": str(duration) + " seconds",
                "Mark": (2 * duration),
                "Link": outputVideo,
                "RId": get_resultId()
            }
            recorded_durations.append(SDViolation)
            write_json(SDViolation)
            reduceBitRate(video[3], outputVideo)
            move_file_to_output_folder(outputVideo)
            terminate_exam(SDViolation["Name"], img)
            
            os.remove(video[3])
            video[3] = str(random.randint(1, 50000)) + ".mp4"
            writer[3] = cv2.VideoWriter(video[3], cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))
            flag[3] = False
            
    elif text != "Stay in the Test" and str(text) == prev_state[3] and (time.time() - start_time[3]) <= 3:
        flag[3] = False
        for _ in range(2):
            writer[3].write(img)
    else:
        if prev_state[3] != "Stay in the Test" and not stop_proctoring_flag:
            writer[3].release()
            end_time[3] = time.time()
            os.remove(video[3])
            video[3] = str(random.randint(1, 50000)) + ".mp4"
            writer[3] = cv2.VideoWriter(video[3], cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))
            flag[3] = False
    prev_state[3] = text

# Function to capture the screen using PyAutoGUI and return the frame as a NumPy array
def capture_screen():
    if pyautogui is None:
        # Return a blank frame if pyautogui is not available
        return np.zeros((1080, 1920, 3), dtype=np.uint8)
    try:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

#Recording Function for Electronic Devices Detection
def EDD_record_duration(text, img):
    global start_time, end_time, prev_state, flag, writer,recorded_Images,EDD_Duration, video, EDWidth, EDHeight
    print(text)
    # Generalized check: If text contains "Detected" (covers "Electronic Device Detected", "Book Detected")
    is_violation = "Detected" in text and "No" not in text
    was_safe = "No" in prev_state[4]
    
    if is_violation and was_safe:
        start_time[4] = time.time()
        for _ in range(2):
            writer[4].write(img)
    elif is_violation and str(text) == prev_state[4] and (time.time() - start_time[4]) > 0:
        flag[4] = True
        for _ in range(2):
            writer[4].write(img)
            
        # Immediate Termination
        if not stop_proctoring_flag:
            writer[4].release()
            end_time[4] = time.time()
            duration = math.ceil((end_time[4] - start_time[4])/10)
            outputVideo = 'EDViolation' + video[4]
            # Save violation evidence image
            img_filename = f"ed_violation_{int(time.time())}.jpg"
            cv2.imwrite(img_filename, img)
            move_file_to_output_folder(img_filename, 'Violations')
            
            EDViolation = {
                "Name": prev_state[4],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[4])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(1.5 * duration),
                "Link": outputVideo,
                "Image": img_filename, # Added Image field
                "RId": get_resultId()
            }
            recorded_durations.append(EDViolation) # Added this missing append in previous logic if it wasn't there
            write_json(EDViolation)
            reduceBitRate(video[4], outputVideo)
            move_file_to_output_folder(outputVideo)
            terminate_exam(EDViolation["Name"], img)
            
            os.remove(video[4])
            video[4]= str(random.randint(1, 50000)) + ".mp4"
            writer[4] = cv2.VideoWriter(video[4], cv2.VideoWriter_fourcc(*'mp4v'), 10 , (EDWidth,EDHeight))
            flag[4] = False

    elif is_violation and str(text) == prev_state[4] and (time.time() - start_time[4]) <= 0:
        flag[4] = False
        for _ in range(2):
            writer[4].write(img)
    else:
        # Violation ended
        if "Detected" in prev_state[4] and "No" not in prev_state[4] and not stop_proctoring_flag:
            writer[4].release()
            end_time[4] = time.time()
            os.remove(video[4])
            video[4]= str(random.randint(1, 50000)) + ".mp4"
            writer[4] = cv2.VideoWriter(video[4], cv2.VideoWriter_fourcc(*'mp4v'), 10 , (EDWidth,EDHeight))
            flag[4] = False
    prev_state[4] = text

#system Related
def deleteTrashVideos():
    global video
    video_folder = 'C:/Users/kaungmyat/PycharmProjects/BestOnlineExamProctor'
    # Iterate through files in the folder
    for filename in os.listdir(video_folder):
        if filename.lower().endswith('.mp4'):
            try:
                os.remove(filename)
            except OSError:
                pass

#Models Related
#One: Face Detection Function
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        if not os.path.exists('static/Profiles'):
            print("Profiles directory not found, creating...")
            os.makedirs('static/Profiles', exist_ok=True)
            
        for image in os.listdir('static/Profiles'):
            # Skip non-image files
            if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                print(f"Processing profile: {image}")
                # Load image with face_recognition (loads as RGB)
                face_image = face_recognition.load_image_file(f"static/Profiles/{image}")
                
                # Check for empty image
                if face_image is None or face_image.size == 0:
                    print(f"Skipping empty image: {image}")
                    continue

                # Verify image type and convert if necessary (ensure 8-bit RGB)
                if face_image.dtype != np.uint8:
                     face_image = face_image.astype(np.uint8)
                
                # Encode face
                encodings = face_recognition.face_encodings(face_image)
                if len(encodings) > 0:
                    face_encoding = encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(image)
                else:
                    print(f"No face found in {image}")
            except Exception as e:
                print(f"Error processing {image}: {e}")
                continue # Ensure we continue to next image!
        print(f"Loaded {len(self.known_face_names)} profiles: {self.known_face_names}")

    def run_recognition(self):
        global Globalflag
        #video_capture = cv2.VideoCapture(0)
        print(f'Face Detection Flag is {Globalflag}')
        text = ""
        if not cap.isOpened():
            sys.exit('Video source not found...')

        while Globalflag and not stop_proctoring_flag:
            frame = get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            
            text = "Verified Student disappeared"
            print("Running Face Verification Function")
            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                #rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        tempname = str(self.known_face_names[best_match_index]).split('_')[0]
                        tempconfidence = face_confidence(face_distances[best_match_index])
                        if tempname == Student_Name and float(tempconfidence[:-1]) >= 84:
                            name = tempname
                            confidence = tempconfidence

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                if "Unknown" not in name:
                    # Create the frame with the name
                    text = "Verified Student appeared"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the resulting image
           # cv2.imshow('Face Recognition', frame)
            print(text)
            faceDetectionRecording(frame, text)
            # Hit 'q' on the keyboard to quit!

#Second: Head Movement Detection Function
def headMovmentDetection(image, face_mesh):
    print("Running HeadMovement Function")
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    try:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)
                
                if len(face_2d) < 4: continue # PnP needs at least 4 points

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                
                if not success: continue

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                
                textHead = ''
                # See where the user's head tilting
                # Sensitivity increased (thresholds lowered)
                if y < -8: # was -10
                    textHead = "Looking Left"
                elif y > 8: # was 15
                    textHead = "Looking Right"
                elif x < -5: # was -8
                    textHead = "Looking Down"
                elif x > 8: # was 15
                    textHead = "Looking Up"
                else:
                    textHead = "Forward"

                # DEBUG overlay for angles
                try:
                    cv2.putText(image, f"X:{int(x)} Y:{int(y)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                except: pass

                # Mouth Analysis (MAR)
                try:
                    # Get full landmarks for MAR
                    lms = face_landmarks.landmark
                    mar = calculate_mar(lms)
                    if mar > 0.4: # Threshold for talking/yawning
                        # textHead = f"Mouth Open ({mar:.2f})" # Don't overwrite Gaze text, just flag?
                        cv2.putText(image, f"Mouth Open ({mar:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    pass
                    
                # Add the text on the image
                cv2.putText(image, textHead, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                Head_record_duration(textHead, image)
    except Exception as e:
        print(f"Error in HeadMovement: {e}")


#Third : More than one person Detection Function
def MTOP_Detection(img):
    print("Running MTOP Function")
    textMTOP = ''
    
    if faceDetection is not None:
        # Use MediaPipe if available
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                # Drawing the recantangle
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                # cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 10)
            if id > 0:
                textMTOP = "More than one person detected."
            else:
                textMTOP = "Only one person is detected"
        else:
            textMTOP="Only one person is detected"
    else:
        # Fallback to OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 1:
            textMTOP = "More than one person detected."
        elif len(faces) == 1:
            textMTOP = "Only one person is detected"
        else:
            textMTOP = "Only one person is detected"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    
    MTOP_record_duration(textMTOP, img)
    print(textMTOP)

#Fourth : Screen Detection Function ( Key-words and Screens)
def shortcut_handler(event):
    if keyboard is None:
        return
    if event.event_type == keyboard.KEY_DOWN:
        shortcut = ''
        # Check for Ctrl+C
        if keyboard.is_pressed('ctrl') and keyboard.is_pressed('c'):
            shortcut += 'Ctrl+C'
            print("Ctrl+C shortcut detected!")
        # Check for Ctrl+V
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('v'):
            shortcut += 'Ctrl+V'
            print("Ctrl+V shortcut detected!")
        # Check for Ctrl+A
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('a'):
            shortcut += 'Ctrl+A'
            print("Ctrl+A shortcut detected!")
        # Check for Ctrl+X
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('x'):
            shortcut += 'Ctrl+X'
            print("Ctrl+X shortcut detected!")
        # Check for Alt+Shift+Tab
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('shift') and keyboard.is_pressed('tab'):
            shortcut += 'Alt+Shift+Tab'
            print("Alt+Shift+Tab shortcut detected!")
        # Check for Win+Tab
        elif keyboard.is_pressed('win') and keyboard.is_pressed('tab'):
            shortcut += 'Win+Tab'
            print("Win+Tab shortcut detected!")
        # Check for Alt+Esc
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('esc'):
            shortcut += 'Alt+Esc'
            print("Alt+Esc shortcut detected!")
        # Check for Alt+Tab
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('tab'):
            shortcut += 'Alt+Tab'
            print("Alt+Tab shortcut detected!")
        # Check for Ctrl+Esc
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('esc'):
            shortcut += 'Ctrl+Esc'
            print("Ctrl+Esc shortcut detected!")
        # Check for Function Keys F1
        elif keyboard.is_pressed('f1'):
            shortcut += 'F1'
            print("F1 shortcut detected")
        # Check for Function Keys F2
        elif keyboard.is_pressed('f2'):
            shortcut += 'F2'
            print("F2 shortcut detected!")
        # Check for Function Keys F3
        elif keyboard.is_pressed('f3'):
            shortcut += 'F3'
            print("F3 shortcut detected!")
        # Check for Window Key
        elif keyboard.is_pressed('win'):
            shortcut += 'Window'
            print("Window shortcut detected!")
        # Check for Ctrl+Alt+Del
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('alt') and keyboard.is_pressed('del'):
            shortcut += 'Ctrl+Alt+Del'
            print("Ctrl+Alt+Del shortcut detected!")
        # Check for Prt Scn
        elif keyboard.is_pressed('print_screen'):
            shortcut += 'Prt Scn'
            print("Prt Scn shortcut detected!")
        # Check for Ctrl+T
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('t'):
            shortcut += 'Ctrl+T'
            print("Ctrl+T shortcut detected!")
        # Check for Ctrl+W
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('w'):
            shortcut += 'Ctrl+W'
            print("Ctrl+W shortcut detected!")
        # Check for Ctrl+Z
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('z'):
            shortcut += 'Ctrl+Z'
            print("Ctrl+Z shortcut detected!")
        
        if shortcut != "":
            shorcuts.append(shortcut)
            global shortcut_flag, shortcut_event_name
            shortcut_flag = True
            shortcut_event_name = shortcut

def screenDetection():
    global active_window, active_window_title, exam_window_title
    if gw is None:
        return
    textScreen = ""
    # Get the current active window
    new_active_window = gw.getActiveWindow()
    frame = capture_screen()

    # Check if the active window has changed
    if new_active_window is not None and new_active_window.title != exam_window_title:
        # Check if the active window is a browser or a tab
        if new_active_window.title != active_window_title:
            print("Moved to Another Window: ", new_active_window.title)
            # Update the active window and its title
            active_window = new_active_window
            active_window_title = active_window.title
        textScreen = "Move away from the Test"
    else:
        if new_active_window is not None:
            textScreen = "Stay in the Test"
    SD_record_duration(textScreen, frame)
    print(textScreen)

#Fifth : Electronic Devices Detection Function
def electronicDevicesDetection(frame):
    global model, EDFlag
    textED = "No Electronic Device Detected"
    
    try:
        # Predict on image with lower confidence threshold for better detection
        # Lowered to 0.25 based on user feedback
        detect_params = model.predict(source=[frame], conf=0.25, save=False, verbose=False)
        
        for result in detect_params:  # iterate results
            boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for box in boxes:  # iterate boxes
                detected_obj = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                # Log all detections for debugging
                # print(f"YOLO Detected: {detected_obj} (confidence: {confidence:.2f})")
                
                # Check for electronic devices and BOOKS
                if detected_obj in ['cell phone', 'remote', 'laptop', 'tv', 'keyboard', 'mouse', 'book']:
                    EDFlag = True
                    textED = f'{detected_obj.title()} Detected'
                    print(f"⚠️ VIOLATION: {detected_obj} detected with {confidence:.2f} confidence!")
                    
                    # Draw Bounding Box for Evidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{detected_obj} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break
            
            if EDFlag:
                break
                
    except Exception as e:
        print(f"Error in electronic device detection: {e}")
    
    # Call recording function BEFORE resetting flag
    EDD_record_duration(textED, frame)
    print(f"Device Detection Status: {textED}")
    
    # Reset flag for next detection cycle
    EDFlag = False

# System Monitoring
def check_prohibited_processes():
    """Check for prohibited processes like Screen Sharing or Remote Desktop."""
    prohibited = ["TeamViewer.exe", "AnyDesk.exe", "Zoom.exe", "Discord.exe", "Skype.exe", "chrome.exe" ] # chrome is tricky as we use it, but maybe remote desktop version?
    # Better list:
    prohibited_apps = ["TeamViewer.exe", "AnyDesk.exe", "Zoom.exe", "webex.exe"]
    
    try:
        # Use tasklist for Windows compatibility without extra deps
        output = subprocess.check_output("tasklist", shell=True).decode()
        for app in prohibited_apps:
            if app.lower() in output.lower():
                print(f"PROHIBITED PROCESS DETECTED: {app}")
                return app
    except Exception as e:
        print(f"Error checking processes: {e}")
    return None

def check_extension_heartbeat():
    """Verify extensions is sending heartbeats."""
    global extension_heartbeat_time
    if extension_heartbeat_time == 0:
        return True # Not started yet
    
    if time.time() - extension_heartbeat_time > 60: # 1 minute timeout
        print("EXTENSION HEARTBEAT LOST!")
        return False
    return True

# MAR Calculation for Mouth Detection
def calculate_mar(lips):
    # lips indices: 
    # Upper: [82, 13, 312] (but usually we take specific points for MAR)
    # Ref: https://github.com/ali-h-khasemi/Mouth-Down-Interval-Detection
    # 6 points: P1(img, 61), P2(img, 291) - corners
    # P3(img, 81), P4(img, 178), P5(img, 311), P6(img, 402) - upper/lower lips
    # Let's use simpler index logic if we get landmarks directly as objects
    
    # Distance function
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    # 4 vertical points pairs, 1 horizontal pair
    # Using mediapipe indices directly (approximate inner lip)
    # UpperInner: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
    # LowerInner: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
    
    # Simple MAR: (p2-p8) + (p3-p7) + (p4-p6) / 2*(p1-p5) ??
    # A simplified approach using upper lip bottom and lower lip top
    # 13 (upper lip mid), 14 (lower lip mid)
    # 78 (left corner), 308 (right corner)
    
    u_mid = lips[13]
    l_mid = lips[14]
    l_corner = lips[78]
    r_corner = lips[308]
    
    vertical = math.sqrt((u_mid.x - l_mid.x)**2 + (u_mid.y - l_mid.y)**2)
    horizontal = math.sqrt((l_corner.x - r_corner.x)**2 + (l_corner.y - r_corner.y)**2)
    
    if horizontal == 0: return 0
    return vertical / horizontal

#Sixth Function : Voice Detection
class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / SHORT_WIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        if pyaudio is None:
            self.p = None
            self.stream = None
            return
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      output=True,
                                      frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.p = None
            self.stream = None
        self.time = time.time()
        self.quiet = []
        self.quiet_idx = -1
        self.timeout = 0

    def record(self):
        global Globalflag
        print('')
        print(f'Voice Flag is {Globalflag}')
        sound = []
        start = time.time()
        begin_time = None
        while Globalflag and not stop_proctoring_flag:
            if self.stream is None:
                time.sleep(1)
                continue
            data = self.stream.read(CHUNK)
            rms_val = self.rms(data)
            if self.inSound(data):
                sound.append(data)
                if begin_time == None:
                    begin_time = datetime.datetime.now()
            else:
                if len(sound) > 0:
                    duration=math.floor((datetime.datetime.now()-begin_time).total_seconds())
                    self.write(sound, begin_time, duration)
                    sound.clear()
                    begin_time = None
                else:
                    self.queueQuiet(data)

            curr = time.time()
            secs = int(curr - start)
            tout = 0 if self.timeout == 0 else int(self.timeout - curr)
            label = 'Listening' if self.timeout == 0 else 'Recording'
            print('[+] %s: Level=[%4.2f] Secs=[%d] Timeout=[%d]' % (label, rms_val, secs, tout), end='\r')

    # quiet is a circular buffer of size cushion
    def queueQuiet(self, data):
        self.quiet_idx += 1
        # start over again on overflow
        if self.quiet_idx == CUSHION_FRAMES:
            self.quiet_idx = 0

        # fill up the queue
        if len(self.quiet) < CUSHION_FRAMES:
            self.quiet.append(data)
        # replace the element on the index in a cicular loop like this 0 -> 1 -> 2 -> 3 -> 0 and so on...
        else:
            self.quiet[self.quiet_idx] = data

    def dequeueQuiet(self, sound):
        if len(self.quiet) == 0:
            return sound

        ret = []

        if len(self.quiet) < CUSHION_FRAMES:
            ret.append(self.quiet)
            ret.extend(sound)
        else:
            ret.extend(self.quiet[self.quiet_idx + 1:])
            ret.extend(self.quiet[:self.quiet_idx + 1])
            ret.extend(sound)

        return ret

    def inSound(self, data):
        rms = self.rms(data)
        curr = time.time()

        if rms > TRIGGER_RMS:
            self.timeout = curr + TIMEOUT_SECS
            return True

        if curr < self.timeout:
            return True

        self.timeout = 0
        return False

    def write(self, sound, begin_time, duration):
        # insert the pre-sound quiet frames into sound
        sound = self.dequeueQuiet(sound)

        # sound ends with TIMEOUT_FRAMES of quiet
        # remove all but CUSHION_FRAMES
        keep_frames = len(sound) - TIMEOUT_FRAMES + CUSHION_FRAMES
        recording = b''.join(sound[0:keep_frames])
        filename = str(random.randint(1,50000))+"VoiceViolation"
        pathname = os.path.join(f_name_directory, '{}.wav'.format(filename))
        wf = wave.open(pathname, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        voiceViolation = {
            "Name": "Common Noise is detected.",
            "Time": begin_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": str(duration) + " seconds",
            "Mark": duration,
            "Link": '{}.wav'.format(filename),
            "RId": get_resultId()
        }
        write_json(voiceViolation)
        print('[+] Saved: {}'.format(pathname))

def cheat_Detection1():
    deleteTrashVideos()
    global Globalflag
    
    # Try to import face_mesh
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            print("Warning: MediaPipe solutions not available, head movement detection disabled")
            face_mesh = None
    except Exception as e:
        print(f"Face mesh initialization error: {e}")
        face_mesh = None
    print(f'CD1 Flag is {Globalflag}')
    while Globalflag:
        image = get_frame()
        if image is None:
            time.sleep(0.05)
            continue
        
        if face_mesh is not None:
            headMovmentDetection(image, face_mesh)
        else:
            # Skip head movement detection if face_mesh is not available
            time.sleep(0.1)
    if Globalflag:
        cap.release()
    deleteTrashVideos()

def cheat_Detection2():
    global Globalflag, shorcuts, no_person_status
    print(f'=== CHEAT DETECTION 2 STARTED === Flag is {Globalflag}')

    deleteTrashVideos()
    frame_count = 0
    skip_frames = 2  # Process every 3rd frame for speed
    
    # Initialize status
    no_person_status['detected'] = True
    no_person_status['start_time'] = None
    
    # Timing Constants for Face Presence (Synced with app.py)
    ALERT_THRESHOLD = 0
    COUNTDOWN_START_THRESHOLD = 0
    TOTAL_TIMEOUT = 30 
    
    while Globalflag:
        image = get_frame()
        if image is None:
            time.sleep(0.05)
            continue
        
        # Skip frames for faster processing
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # Resize for faster processing (reduce to 640x480)
        try:
            image_resized = cv2.resize(image, (640, 480))
            # Lighter brightness enhancement (faster)
            image_processed = cv2.convertScaleAbs(image_resized, alpha=1.15, beta=25)
        except Exception:
            # Fallback if resize fails
            image_processed = image.copy()
        
        if frame_count % 60 == 0:  # Log every 60 frames (~2 seconds)
            print(f"Processing frame {frame_count} - Detection active (Optimized)")
        
        image1 = image_processed.copy()
        
        # Check for person detection first
        person_detected = check_person_present(image1)
        
        if not person_detected:
            no_person_status['detected'] = False
            if no_person_status['start_time'] is None:
                no_person_status['start_time'] = time.time()
                print("⚠️ WARNING: No person detected in frame!")
            else:
                elapsed = time.time() - no_person_status['start_time']
                
                # Check for termination
                if elapsed >= TOTAL_TIMEOUT:
                    print(f"❌ TERMINATING: No person detected for {TOTAL_TIMEOUT} seconds!")
                    # Pass the original image (or optimized one) as evidence of empty seat
                    terminate_exam("No participant detected in camera feed", image1)
                    break
        else:
            # Reset timer if person is detected
            if no_person_status['start_time'] is not None:
                print("✅ Person detected again - timer reset")
            no_person_status['detected'] = True
            no_person_status['start_time'] = None
        
        # Run other detections
        MTOP_Detection(image1)
        screenDetection()
        electronicDevicesDetection(image1.copy())
        
        if shortcut_flag:
            print(f"SHORTCUT DETECTED: {shortcut_event_name}")
            Shortcut_record_duration(f"Shortcut ({shortcut_event_name}) detected", image)
            shortcut_flag = False
            
        # System Monitoring Checks (Every ~5 seconds efficiently)
        if frame_count % 100 == 0:
            bad_app = check_prohibited_processes()
            if bad_app:
                print(f"⚠️ PROHIBITED APP: {bad_app}")
                terminate_exam(f"Prohibited Application Detected: {bad_app}", image1)
                break
            
            if not check_extension_heartbeat():
                 print("⚠️ BROWSER EXTENSION DISABLED")
                 # terminate_exam("Browser Extension Disabled or Removed", image1)
                 # warn first?
    
    print("=== CHEAT DETECTION 2 STOPPED ===")
    deleteTrashVideos()
    if cap is not None:
        cap.release()

def check_person_present(image):
    """Quick check if a person/face is present in the frame."""
    try:
        # Use local Haar Cascade file which is known to exist
        cascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
             # Fallback to cv2 data if local not found (though mapped file suggests it is there)
             cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
             
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0
    except Exception as e:
        print(f"Error in person detection: {e}")
        return True  # Assume person present on error to avoid false termination

#Query Related
#Function to give the next resut id
def get_resultId():
    return resultId

def get_suspicion_log():
    with open('suspicion_log.json','r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        return file_data

#Function to give the trust score
def get_TrustScore(Rid):
    with open('violation.json', 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        filtered_data = [item for item in file_data if item["RId"] == Rid]
        total_mark = sum(item["Mark"] for item in filtered_data)
        return total_mark

#Function to give all results
def getResults():
    with open('result.json', 'r+') as file:
        # First we load existing data into a dict.
        result_data = json.load(file)
        return result_data

#Function to give result details
def getResultDetails(rid):
    with open('result.json', 'r+') as file:
        # First we load existing data into a dict.
        result_data = json.load(file)
        filtered_result = [item for item in result_data if item["Id"] == int(rid)]
    with open('violation.json', 'r+') as file:
        # First we load existing data into a dict.
        violation_data = json.load(file)
        filtered_violations = [item for item in violation_data if item["RId"] == int(rid)]
    resultDetails = {
            "Result": filtered_result,
            "Violation": filtered_violations
        }
    return resultDetails

# Start the master frame reader automatically in a background thread
master_reader_thread = threading.Thread(target=master_frame_reader, daemon=True)
master_reader_thread.start()

a = Recorder()
fr = FaceRecognition()
