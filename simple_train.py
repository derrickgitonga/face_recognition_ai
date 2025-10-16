import cv2
import os
import time

def collect_face_samples():
    """Simple tool to collect face images for training"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("=== Face Sample Collection ===")
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
    print(f"\nCollecting images for {person_name}...")
    print("Press 'c' to capture, 'q' to quit")
    print("Make different expressions and angles")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"Collecting faces for {person_name} - Press 'c' to capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            # Save all detected faces
            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                filename = f"{person_dir}/face_{count}_{i}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"Saved: {filename}")
            count += len(faces)
            print(f"Total captured: {count}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCompleted! Collected {count} face images for {person_name}")
    print(f"Images saved in: {person_dir}")

if __name__ == "__main__":
    collect_face_samples()