"""
Functional test script for Voice/Noise Detection.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add existing directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies
mock_cv2 = MagicMock()
mock_cv2.__version__ = '4.0.0'
sys.modules['cv2'] = mock_cv2

mock_mediapipe = MagicMock()
sys.modules['mediapipe'] = mock_mediapipe

mock_face_recognition = MagicMock()
sys.modules['face_recognition'] = mock_face_recognition

# Mock PyAudio BEFORE utils import
mock_pyaudio_pkg = MagicMock()
sys.modules['pyaudio'] = mock_pyaudio_pkg

mock_ultralytics = MagicMock()
mock_ultralytics.__version__ = '8.0.0'
sys.modules['ultralytics'] = mock_ultralytics

import utils

class TestVoiceDetection(unittest.TestCase):
    
    def setUp(self):
        # Reset state
        utils.Globalflag = True
        utils.stop_proctoring_flag = False
        
        # Instantiate Recorder
        # We need to mock the stream within proper context
        # The Recorder.__init__ creates self.p = pyaudio.PyAudio() and self.stream
        
        # Mock pyaudio.PyAudio() instance
        self.mock_pa_instance = MagicMock()
        mock_pyaudio_pkg.PyAudio.return_value = self.mock_pa_instance
        
        # Mock stream
        self.mock_stream = MagicMock()
        self.mock_pa_instance.open.return_value = self.mock_stream
        
        self.recorder = utils.Recorder()
        
    def test_noise_detected_triggers_write(self):
        print("\n🧪 Testing Noise Detection...")
        
        # Mock rms logic: 'rms' is a method in Recorder
        # We want to simulate high RMS > TRIGGER_RMS (which is 10)
        
        # Wait, utils.Recorder.record() calls self.stream.read(CHUNK)
        # Then self.rms(data)
        # Then self.inSound(data) -> checks rms > TRIGGER_RMS
        
        # Mock stream.read to return dummy bytes
        self.mock_stream.read.return_value = b'\x00' * utils.Recorder.CHUNK
        
        # Mock wave module to prevent actual file writing
        mock_wave = MagicMock()
        sys.modules['wave'] = mock_wave
        
        # Patch the 'rms' method to return a HIGH value
        with patch.object(utils.Recorder, 'rms', return_value=50.0): # 50 > 10
             # Patch 'write' logic wrapper? No, let's test that write() IS called which calls wave.open
             
             # We need to act as if we are running record()
             
             # Patch time.time to simulate passage of time?
             # record() uses time.time()
             
             # Let's simplify: call record() with Short timeout.
             self.recorder.timeout = 1
             
             # Mock os.path.exists/makedirs just in case
             with patch('os.makedirs'), patch('os.path.exists', return_value=True):
                 
                 # Mock inSound to trigger the recording logic: [Silence, Noise, Noise, Silence]
                 # first Silence -> queueQuiet
                 # second Noise -> begin_time set, append sound
                 # third Noise -> append sound
                 # fourth Silence -> write called because sound > 0
                 with patch.object(utils.Recorder, 'inSound', side_effect=[False, True, True, False, False]):
                     self.recorder.record()
                     
                     # Check if wave.open was called (signifying a write attempt)
                     # or check if our mocked wave.open was called
                     mock_wave.open.assert_called()
                     print("✅ Recorder attempted to save audio (wave.open called)")

if __name__ == '__main__':
    unittest.main()
