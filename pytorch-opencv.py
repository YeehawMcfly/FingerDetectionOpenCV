import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def preprocess_image(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # Define transforms to match training exactly
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Changed to 32x32
        transforms.Grayscale(),       # Convert to grayscale
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    # Apply transforms and add batch dimension
    img = transform(img).unsqueeze(0)
    
    return img

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Initialize model
    model = NeuralNetwork().to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load('new-model.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize labels (0-5)
    labels = [str(i) for i in range(6)]
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    roi_size = 300
    
    # Initialize prediction smoothing
    prediction_history = []
    smooth_window = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates
        x = width // 2 - roi_size // 2
        y = height // 2 - roi_size // 2
        
        # Extract and process ROI
        roi = frame[y:y + roi_size, x:x + roi_size]
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
        
        # Get most common prediction in history
        if prediction_history:
            from statistics import mode
            smoothed_prediction = mode(prediction_history)
            current_label = labels[smoothed_prediction]
        else:
            current_label = labels[predicted_class]
        
        # Only show high confidence predictions
        if confidence > 0.7:
            text_color = (0, 255, 0)  # Green for high confidence
        else:
            text_color = (0, 165, 255)  # Orange for low confidence
            
        # Display prediction and confidence
        text = f"Number: {current_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Draw ROI rectangle with visual guide
        cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
        
        # Add visual guides
        center_x = x + roi_size // 2
        center_y = y + roi_size // 2
        cv2.line(frame, (center_x, y), (center_x, y + roi_size), (0, 255, 0), 1)
        cv2.line(frame, (x, center_y), (x + roi_size, center_y), (0, 255, 0), 1)
        
        # Add instruction text
        cv2.putText(frame, "Place hand in box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Finger Number Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()