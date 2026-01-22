"""
Functional test script for Screen/Window Detection.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add existing directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies matching test_mobile_detection.py setup
mock_cv2 = MagicMock()
mock_cv2.__version__ = '4.0.0'
sys.modules['cv2'] = mock_cv2

mock_mediapipe = MagicMock()
sys.modules['mediapipe'] = mock_mediapipe

mock_face_recognition = MagicMock()
sys.modules['face_recognition'] = mock_face_recognition

mock_pyaudio = MagicMock()
sys.modules['pyaudio'] = mock_pyaudio

mock_ultralytics = MagicMock()
mock_ultralytics.__version__ = '8.0.0'
sys.modules['ultralytics'] = mock_ultralytics

# Mock pyautogui and pygetwindow BEFORE importing utils
mock_pyautogui = MagicMock()
sys.modules['pyautogui'] = mock_pyautogui

mock_gw = MagicMock()
sys.modules['pygetwindow'] = mock_gw

import utils

class TestScreenDetection(unittest.TestCase):
    
    def setUp(self):
        # Reset global state
        utils.active_window_title = "Exam Window"
        utils.exam_window_title = "Exam Window"
        utils.writer = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()] # Mock video writers
        utils.Globalflag = True
        utils.stop_proctoring_flag = False
        
    def test_switch_window_triggers_violation(self):
        print("\n🧪 Testing Screen Detection (Window Switch)...")
        
        # Mock active window to be DIFFERENT from exam title
        mock_window = MagicMock()
        mock_window.title = "Cheat Sheet - Notepad"
        
        # Mock gw.getActiveWindow to return our bad window
        utils.gw.getActiveWindow.return_value = mock_window
        
        # Mock capture_screen to return a dummy frame
        with patch('utils.capture_screen', return_value=np.zeros((100,100,3), dtype=np.uint8)):
            # Mock SD_record_duration to intercept the call
            with patch('utils.SD_record_duration') as mock_record:
                
                # Run detection
                utils.screenDetection()
                
                # Verify it detected the switch
                # It should call SD_record_duration with "Move away from the Test"
                mock_record.assert_called_with("Move away from the Test", unittest.mock.ANY)
                print("✅ Correctly flagged 'Move away from the Test'")
                
                # Verify it updated the active_window_title
                self.assertEqual(utils.active_window_title, "Cheat Sheet - Notepad")

    def test_same_window_is_safe(self):
        print("\n🧪 Testing Safe Window State...")
        
        # Reset title
        utils.active_window_title = "Exam Window"
        utils.exam_window_title = "Exam Window"
        
        mock_window = MagicMock()
        mock_window.title = "Exam Window"
        utils.gw.getActiveWindow.return_value = mock_window
        
        with patch('utils.capture_screen', return_value=np.zeros((100,100,3), dtype=np.uint8)):
            with patch('utils.SD_record_duration') as mock_record:
                utils.screenDetection()
                
                # It calls it with "Stay in the Test"
                mock_record.assert_called_with("Stay in the Test", unittest.mock.ANY)
                print("✅ Correctly confirmed 'Stay in the Test'")

if __name__ == '__main__':
    unittest.main()
