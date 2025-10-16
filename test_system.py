import cv2
import os
import time

def test_camera_and_detection():
    """Simple test to verify camera and face detection work"""
    print("=== Testing Camera and Face Detection ===")
    
    # Test camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Test face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Capture a few frames and test detection
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture frame {i}")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        print(f"Frame {i+1}: Detected {len(faces)} faces")
        
        if len(faces) > 0:
            print("✅ Face detection working!")
            cap.release()
            return True
        
        time.sleep(0.5)
    
    cap.release()
    print("❌ No faces detected in test frames")
    return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = ['datasets', 'unknown_faces', 'output']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")

if __name__ == "__main__":
    print("System Diagnostic Check\n")
    check_directories()
    print("\nTesting camera and face detection...")
    test_camera_and_detection()