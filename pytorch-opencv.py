import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from statistics import mode
import sys

# Define the model architecture (match the .ipynb)
class ConvNeuralNetwork(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

def preprocess_image(img):
    """Preprocess image to match training pipeline."""
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # Define transforms to match training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Apply transforms and add batch dimension
    img = transform(img).unsqueeze(0)
    return img

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Initialize model
    model = ConvNeuralNetwork().to(device)
    
    # Load model weights
    model_path = "model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize labels
    labels = [str(i) for i in range(6)]
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    # Define parameters
    roi_size = 300
    target_size = 32
    smooth_window = 5
    confidence_threshold = 0.7
    prediction_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Calculate ROI coordinates
            x = width // 2 - roi_size // 2
            y = height // 2 - roi_size // 2
            
            # Validate and extract ROI
            if x < 0 or y < 0 or x + roi_size > width or y + roi_size > height:
                print("Warning: Invalid ROI coordinates")
                continue
            roi = frame[y:y + roi_size, x:x + roi_size]
            if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                print("Warning: ROI size mismatch")
                continue
            
            # Preprocess ROI
            processed_roi = preprocess_image(roi)
            
            # Make prediction
            with torch.no_grad():
                processed_roi = processed_roi.to(device)
                outputs = model(processed_roi)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities[0]).item()
                confidence = float(probabilities[0][predicted_class])
            
            # Smooth predictions
            prediction_history.append(predicted_class)
            if len(prediction_history) > smooth_window:
                prediction_history.pop(0)
            
            # Get smoothed prediction
            smoothed_prediction = mode(prediction_history) if prediction_history else predicted_class
            current_label = labels[smoothed_prediction]
            
            # Set text color based on confidence
            text_color = (0, 255, 0) if confidence > confidence_threshold else (0, 165, 255)
            
            # Display prediction
            text = f"Number: {current_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            # Draw ROI and guides
            cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
            center_x, center_y = x + roi_size // 2, y + roi_size // 2
            cv2.line(frame, (center_x, y), (center_x, y + roi_size), (0, 255, 0), 1)
            cv2.line(frame, (x, center_y), (x + roi_size, center_y), (0, 255, 0), 1)
            cv2.putText(frame, "Place hand in box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Finger Number Recognition', frame)
            
            # Control frame rate and check for quit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Stream interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()