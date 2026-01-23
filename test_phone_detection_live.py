"""
Test Mobile Phone Detection
Run this to test if the model can detect phones correctly
"""

from ultralytics import YOLO
import cv2
import numpy as np

def test_phone_detection():
    print("=" * 60)
    print("🔬 MOBILE PHONE DETECTION TEST")
    print("=" * 60)
    
    # Load YOLOv8n model
    print("\n📦 Loading YOLOv8n model...")
    try:
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check available classes
    print(f"\n📋 Model has {len(model.names)} classes")
    print("Phone-related classes:")
    for class_id, class_name in model.names.items():
        if 'phone' in class_name.lower() or 'cell' in class_name.lower():
            print(f"  ✅ ID {class_id}: {class_name}")
    
    # Test with dummy frame
    print("\n🧪 Testing with dummy frame...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        results = model.predict(source=[dummy_frame], conf=0.20, verbose=False)
        print("✅ Model inference works")
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return
    
    # Test with webcam
    print("\n📹 Testing with live webcam...")
    print("Press 'q' to quit")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    frame_count = 0
    detected_phones = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model.predict(source=[frame], conf=0.20, save=False, verbose=False)
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                detected_obj = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                # Draw detection
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if detected_obj == 'cell phone':
                    detected_phones += 1
                    color = (0, 0, 255)  # Red for phone
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"📱 PHONE {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    print(f"✅ Phone detected! Confidence: {confidence:.2f}")
                else:
                    # Other objects in gray
                    color = (100, 100, 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, f"{detected_obj}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display info
        info = f"Frame: {frame_count} | Phones detected: {detected_phones} | Conf threshold: 0.20"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Phone Detection Test', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("-" * 60)
    print(f"✅ Test complete!")
    print(f"📊 Frames processed: {frame_count}")
    print(f"📱 Phones detected: {detected_phones}")
    if frame_count > 0:
        print(f"📈 Detection rate: {(detected_phones/frame_count)*100:.1f}%")

if __name__ == "__main__":
    print("\n")
    test_phone_detection()
    print("\n")
