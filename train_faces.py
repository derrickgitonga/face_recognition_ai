import cv2
import os
import numpy as np
import pickle

def collect_training_images():
    """Collect face images for training"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("[INFO] Enter person's name for training: ")
    person_name = input().strip()
    
    # Create directory for this person
    person_dir = f"training_data/{person_name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    count = 0
    max_images = 50  # Collect 50 images
    
    print(f"[INFO] Collecting {max_images} images for {person_name}")
    print("[INFO] Look at the camera and move your head slightly")
    print("[INFO] Press 'q' to stop early")
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Save face
            face_img = gray[y:y+h, x:x+w]
            
            # Resize to consistent size
            face_img = cv2.resize(face_img, (200, 200))
            
            # Save image
            filename = f"{person_dir}/{count}.jpg"
            cv2.imwrite(filename, face_img)
            count += 1
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{max_images}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Small delay
            cv2.waitKey(100)
        
        cv2.imshow("Collecting Training Data", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {count} images for {person_name}")

def train_recognizer():
    """Train the face recognizer with collected data"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_names = []
    label_id = 0
    
    if not os.path.exists("training_data"):
        print("[ERROR] No training data found. Run collect_training_images first.")
        return
    
    # Load training data
    for person_name in os.listdir("training_data"):
        person_dir = os.path.join("training_data", person_name)
        if os.path.isdir(person_dir):
            label_names.append(person_name)
            
            for image_file in os.listdir(person_dir):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(person_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None:
                        faces.append(image)
                        labels.append(label_id)
            
            label_id += 1
    
    if not faces:
        print("[ERROR] No faces found for training")
        return
    
    # Train recognizer
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    recognizer.save("face_model.yml")
    
    # Save label names
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_names, f)
    
    print(f"[SUCCESS] Trained on {len(faces)} faces from {len(label_names)} persons")
    print("[INFO] Model saved as 'face_model.yml'")

if __name__ == "__main__":
    print("1. Collect training images")
    print("2. Train recognizer")
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        collect_training_images()
    elif choice == "2":
        train_recognizer()
    else:
        print("Invalid choice")