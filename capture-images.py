import cv2
import numpy as np
import sys
import time
import os

def resize_frame(frame, width, height, interpolation=cv2.INTER_LINEAR):
    """
    Resize the frame to the specified dimensions.
    """
    return cv2.resize(frame, (width, height), interpolation=interpolation)

def start_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit()

    # Create directories for each label (0-5)
    base_dir = "captured_dataset"
    labels = ['0', '1', '2', '3', '4', '5']
    for label in labels:
        label_dir = os.path.join(base_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
    
    print("Press a number (0-5) to capture frames or 'q' to quit.")
    
    # Define ROI size and display size
    roi_size = 300  # Size of the box we'll show to user
    target_size = 32  # Final size for saving
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            # Flip the frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get the center ROI coordinates
            height, width = frame.shape[:2]
            x = width // 2 - roi_size // 2
            y = height // 2 - roi_size // 2
            
            # Draw ROI rectangle
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
            
            # Add crosshair guides
            center_x = x + roi_size // 2
            center_y = y + roi_size // 2
            cv2.line(display_frame, (center_x, y), (center_x, y + roi_size), (0, 255, 0), 1)
            cv2.line(display_frame, (x, center_y), (x + roi_size, center_y), (0, 255, 0), 1)
            
            # Add instruction text
            cv2.putText(display_frame, "Place hand in box", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Webcam Stream', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                # Extract ROI
                roi = frame[y:y + roi_size, x:x + roi_size]
                
                # Resize ROI to target size
                small_frame = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
                # Save frame with the pressed number as the label
                label = chr(key)
                timestamp = int(time.time() * 1000)  # Millisecond precision
                filename = f"{label}_{timestamp}.png"
                filepath = os.path.join(base_dir, label, filename)
                
                # Save the small frame
                cv2.imwrite(filepath, small_frame)
                print(f"Captured frame saved as {filepath} ({target_size}x{target_size})")
    
    except KeyboardInterrupt:
        print("Stream interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting webcam stream with label-based dataset creation...")
    start_webcam()