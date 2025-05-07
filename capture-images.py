import cv2
import numpy as np
import sys
import time
import os
from datetime import datetime

def resize_frame(frame, width, height, interpolation=cv2.INTER_AREA):
    """Resize the frame to the specified dimensions."""
    return cv2.resize(frame, (width, height), interpolation=interpolation)

def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    # Create directories for each label (0-5)
    base_dir = "captured_dataset"
    labels = ['0', '1', '2', '3', '4', '5']
    for label in labels:
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

    print("Press 0-5 to capture frames or 'q' to quit. Press 'p' to preview last capture.")

    # Define sizes
    roi_size = 300  # Display ROI size
    target_size = 32  # Saved image size
    last_capture = None  # Store last captured frame for preview

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            x = width // 2 - roi_size // 2
            y = height // 2 - roi_size // 2

            # Draw ROI and guides
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
            center_x, center_y = x + roi_size // 2, y + roi_size // 2
            cv2.line(display_frame, (center_x, y), (center_x, y + roi_size), (0, 255, 0), 1)
            cv2.line(display_frame, (x, center_y), (x + roi_size, center_y), (0, 255, 0), 1)
            cv2.putText(display_frame, "Place hand in box", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Webcam Stream', display_frame)

            # Handle input
            key = cv2.waitKey(30) & 0xFF  # Add slight delay for frame rate control
            if key == ord('q'):
                break
            elif key == ord('p') and last_capture is not None:
                cv2.imshow('Last Capture (32x32)', last_capture)
                cv2.waitKey(0)  # Wait for any key to close preview
                cv2.destroyWindow('Last Capture (32x32)')
            elif key in [ord(str(i)) for i in range(6)]:
                roi = frame[y:y + roi_size, x:x + roi_size]
                if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                    print("Error: Invalid ROI size")
                    continue

                small_frame = resize_frame(roi, target_size, target_size)
                label = chr(key)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label}_{timestamp}.png"
                filepath = os.path.join(base_dir, label, filename)
                cv2.imwrite(filepath, small_frame)
                print(f"Captured frame saved as {filepath} ({target_size}x{target_size})")
                last_capture = small_frame
            elif key != 255:  # Ignore invalid keys
                print(f"Invalid key pressed: {chr(key)}. Use 0-5, 'q', or 'p'.")

    except KeyboardInterrupt:
        print("Stream interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting webcam stream with label-based dataset creation...")
    start_webcam()