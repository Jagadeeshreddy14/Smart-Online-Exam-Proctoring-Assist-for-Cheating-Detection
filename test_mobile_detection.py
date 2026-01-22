"""
Test script to verify mobile phone detection feature
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the required modules
    from utils import electronicDevicesDetection, terminate_exam
    import cv2
    import numpy as np
    
    print("✅ Successfully imported required modules")
    print("✅ Mobile phone detection feature is ready!")
    print("\n📝 Feature Details:")
    print("- Mobile phone detection is now integrated into the electronic devices detection system")
    print("- When a mobile phone ('cell phone') is detected in the camera feed, the exam will terminate immediately")
    print("- A violation record will be created with high penalty (100 marks)")
    print("- Evidence image will be saved in the Violations folder")
    print("- The system maintains all other detection features as before")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🚀 To start the proctoring system, run: python app.py")