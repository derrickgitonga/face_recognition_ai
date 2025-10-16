import cv2
import time

def simple_face_test():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    print("Simple Face Detection Test")
    print("Looking for faces for 10 seconds...")
    print("Make sure you're in front of the camera!")
    
    start_time = time.time()
    frames_processed = 0
    faces_detected = 0
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        frames_processed += 1
        if len(faces) > 0:
            faces_detected += len(faces)
            print(f"Frame {frames_processed}: Detected {len(faces)} faces!")
        
        time.sleep(0.1)
    
    cap.release()
    
    print(f"\n=== RESULTS ===")
    print(f"Frames processed: {frames_processed}")
    print(f"Total faces detected: {faces_detected}")
    print(f"Detection rate: {(faces_detected/frames_processed*100):.1f}%")

if __name__ == "__main__":
    simple_face_test()