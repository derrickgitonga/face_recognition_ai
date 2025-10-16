import cv2
import os
import pickle
import face_recognition
from imutils import paths
import argparse

def create_dataset_structure():
    """Create the necessary folder structure"""
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("unknown_faces", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    print("[INFO] Created directory structure")

def encode_faces(dataset_path, encodings_path, detection_method="hog"):
    """
    Encode faces from the dataset directory
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path '{dataset_path}' does not exist")
        print("[INFO] Create folders like: datasets/person_name/image1.jpg")
        return False

    # Get the paths to the images
    image_paths = list(paths.list_images(dataset_path))
    
    if not image_paths:
        print(f"[ERROR] No images found in '{dataset_path}'")
        print("[INFO] Add images to datasets/person_name/ folders")
        return False

    print(f"[INFO] Found {len(image_paths)} images in dataset")
    
    known_encodings = []
    known_names = []
    
    # Loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}: {image_path}")
        
        # Extract the person name from the image path
        name = image_path.split(os.path.sep)[-2]
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not load image: {image_path}")
            continue
            
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        
        if not boxes:
            print(f"[WARNING] No faces detected in {image_path}")
            continue
            
        # Compute facial embeddings
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # Loop over the encodings
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
    
    if not known_encodings:
        print("[ERROR] No face encodings were generated. Check your dataset.")
        return False
    
    # Save the encodings to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(encodings_path, "wb") as f:
        f.write(pickle.dumps(data))
    
    print(f"[SUCCESS] Encoded {len(known_encodings)} faces from {len(set(known_names))} persons")
    print(f"[INFO] Encodings saved to: {encodings_path}")
    return True

def main():
    # Create argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", default="datasets", 
                   help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", default="encodings.pickle", 
                   help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection-method", type=str, default="hog", 
                   help="face detection model to use: either `hog` or `cnn`")
    
    args = vars(ap.parse_args())
    
    # Create directory structure
    create_dataset_structure()
    
    # Encode faces
    success = encode_faces(args["dataset"], args["encodings"], args["detection_method"])
    
    if success:
        print("\n[INFO] Next step: Run 'python face_detection.py' to start recognition")
    else:
        print("\n[ERROR] Failed to encode faces. Please check your dataset.")

if __name__ == "__main__":
    main()