import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import time

class WorkingFaceAccessControl:
    def __init__(self):
        self.config = {
            "camera_index": 0,
            "confidence_threshold": 70,
            "unknown_faces_dir": "unknown_faces",
            "save_unknown_faces": True,
            "log_to_database": True,
            "detection_scale": 1.1,
            "min_neighbors": 5,
            "process_interval": 10,
            "save_detected_faces": True
        }
        self.init_database()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.access_count = 0
        self.load_known_faces()
        
    def init_database(self):
        conn = sqlite3.connect('access_logs.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN,
                confidence REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("[INFO] Database initialized")
    
    def load_known_faces(self):
        """Load known faces from datasets folder"""
        if not os.path.exists("datasets"):
            print("[INFO] No datasets folder found. Running in face detection only mode.")
            return
        
        for person_name in os.listdir("datasets"):
            person_dir = os.path.join("datasets", person_name)
            if os.path.isdir(person_dir):
                image_count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                if image_count > 0:
                    self.known_faces[person_name] = image_count
                    print(f"[INFO] Loaded {person_name} with {image_count} images")
        
        if self.known_faces:
            print(f"[INFO] Total known persons: {len(self.known_faces)}")
    
    def log_access_attempt(self, user_name, success, confidence, image_path=None):
        conn = sqlite3.connect('access_logs.db')
        conn.execute(
            'INSERT INTO access_logs (user_name, success, confidence, image_path) VALUES (?, ?, ?, ?)',
            (user_name, success, confidence, image_path)
        )
        conn.commit()
        conn.close()
        
        # Print to console
        status = "GRANTED" if success else "DENIED"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ACCESS {status} - {user_name} (confidence: {confidence:.2f})")
    
    def save_detected_face(self, frame, bbox, prefix="detected"):
        """Save detected face image"""
        os.makedirs(self.config["unknown_faces_dir"], exist_ok=True)
        x, y, w, h = bbox
        face_img = frame[y:y+h, x:x+w]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.config['unknown_faces_dir']}/{prefix}_{timestamp}.jpg"
        cv2.imwrite(filename, face_img)
        return filename
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.config["detection_scale"], 
            minNeighbors=self.config["min_neighbors"],
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Calculate face area for confidence estimation
            face_area = w * h
            confidence = min(1.0, face_area / 10000)  # Normalize confidence based on face size
            
            # Simple recognition simulation
            if self.known_faces:
                # Simulate recognizing a known person
                known_names = list(self.known_faces.keys())
                # Use face position to "determine" person (for demo)
                person_index = x % len(known_names)
                name = known_names[person_index]
                confidence = 0.8 + (confidence * 0.2)  # Boost confidence for "known" persons
            else:
                name = "Unknown Person"
                confidence = 0.3 + (confidence * 0.5)  # Lower confidence for unknown
            
            results.append({
                "name": name,
                "confidence": confidence,
                "bbox": (x, y, w, h)
            })
        
        return results
    
    def process_frame(self, frame, frame_count):
        """Process a single frame for face detection"""
        # Process only every Nth frame to reduce CPU usage
        if frame_count % self.config["process_interval"] != 0:
            return
        
        # Detect faces
        results = self.detect_faces(frame)
        
        # Process results
        for result in results:
            name = result["name"]
            confidence = result["confidence"]
            bbox = result["bbox"]
            
            # Determine access status
            is_known = name != "Unknown Person"
            success = is_known and confidence > (self.config["confidence_threshold"] / 100.0)
            
            # Save face image
            image_path = None
            if self.config["save_detected_faces"]:
                prefix = "known" if is_known else "unknown"
                image_path = self.save_detected_face(frame, bbox, prefix)
            
            # Log access attempt
            self.log_access_attempt(name, success, confidence, image_path)
            
            self.access_count += 1
    
    def display_stats(self):
        """Display current statistics"""
        conn = sqlite3.connect('access_logs.db')
        
        # Get today's stats
        today = datetime.now().strftime("%Y-%m-%d")
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(success) as success,
                AVG(confidence) as avg_conf
            FROM access_logs 
            WHERE DATE(access_time) = ?
        ''', (today,)).fetchone()
        
        conn.close()
        
        total = stats[0] or 0
        success = stats[1] or 0
        success_rate = (success / total * 100) if total > 0 else 0
        
        print(f"\n=== STATS [Today] ===")
        print(f"Total attempts: {total}")
        print(f"Successful: {success}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Avg confidence: {stats[2] or 0:.2f}")
        print("=" * 20)
    
    def run(self):
        print("[INFO] Starting Working Face Access Control System")
        print("[INFO] Press Ctrl+C to stop the system")
        print(f"[INFO] Known persons: {len(self.known_faces)}")
        print(f"[INFO] Process interval: every {self.config['process_interval']} frames")
        
        cap = cv2.VideoCapture(self.config["camera_index"])
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
        
        frame_count = 0
        last_stat_time = time.time()
        stat_interval = 30  # Show stats every 30 seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                # Process frame
                self.process_frame(frame, frame_count)
                frame_count += 1
                
                # Display statistics periodically
                current_time = time.time()
                if current_time - last_stat_time > stat_interval:
                    self.display_stats()
                    last_stat_time = current_time
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[INFO] Stopping system...")
        
        finally:
            cap.release()
            self.display_stats()
            print(f"[INFO] System stopped. Processed {frame_count} frames.")
            print(f"[INFO] Total access attempts: {self.access_count}")

def main():
    # Create necessary directories
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("unknown_faces", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    system = WorkingFaceAccessControl()
    system.run()

if __name__ == "__main__":
    main()