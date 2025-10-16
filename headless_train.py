import cv2
import os
import time
from datetime import datetime

def headless_face_collection():
    """Collect face images without GUI display"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("=== Headless Face Collection ===")
    person_name = input("Enter person's name: ").strip()
    
    # Create directory
    person_dir = f"datasets/{person_name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    count = 0
    target_count = 50
    print(f"\nCollecting {target_count} images for {person_name}...")
    print("Making different expressions and angles...")
    print("Press Ctrl+C to stop early")
    
    last_capture_time = 0
    capture_interval = 0.5  # Capture every 0.5 seconds
    
    try:
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Only process if enough time has passed since last capture
            if current_time - last_capture_time > capture_interval:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Save all detected faces
                    for i, (x, y, w, h) in enumerate(faces):
                        if count >= target_count:
                            break
                            
                        face_img = frame[y:y+h, x:x+w]
                        filename = f"{person_dir}/face_{count:03d}.jpg"
                        cv2.imwrite(filename, face_img)
                        count += 1
                        last_capture_time = current_time
                        
                        print(f"Captured: {count}/{target_count} - {filename}")
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping collection...")
    
    finally:
        cap.release()
        print(f"\nCompleted! Collected {count} face images for {person_name}")
        print(f"Images saved in: {person_dir}")

if __name__ == "__main__":
    headless_face_collection()