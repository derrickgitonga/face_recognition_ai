import cv2

def test_camera():
    """Test if camera is working"""
    print("Testing camera...")
    
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            
            ret, frame = cap.read()
            if ret:
                print(f"Successfully captured frame from camera {i}")
                cv2.imshow(f"Camera {i} Test - Press any key to close", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Could not read frame from camera {i}")
            
            cap.release()
        else:
            print(f"No camera at index {i}")

if __name__ == "__main__":
    test_camera()