# Hand Gesture Recognition with PyTorch and OpenCV

This repository contains a hand gesture recognition system that classifies hand gestures (numbers 0–5) using a convolutional neural network (CNN) implemented in PyTorch and real-time webcam input processed with OpenCV. The project includes scripts for data capture, model training, and real-time inference, along with a pre-trained model.

## Features
- Captures hand gesture images using a webcam (`capture-images.py`).
- Trains a CNN model on a custom dataset (`train_model.ipynb`).
- Performs real-time gesture recognition with webcam input (`pytorch-opencv.py`).
- Uses a three-way dataset split (train/validation/test) for robust evaluation.
- Includes data augmentation and early stopping for improved training.
- Pre-trained model (`model.pth`) for immediate inference.

## Repository Contents
- `train_model.ipynb`: Jupyter Notebook to train the CNN model on a custom dataset.
- `capture-images.py`: Script to capture hand gesture images via webcam.
- `pytorch-opencv.py`: Script for real-time gesture recognition using the trained model.
- `model.pth`: Pre-trained model weights for the CNN (6 classes, 32x32 grayscale input).

**Note**: The `captured_dataset` folder (containing training/test images) is not included. You must create this folder and populate it with your own images (see [Data Collection](#data-collection)).

## Prerequisites
- **Python**: 3.8 or higher
- **Libraries**:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`
  - `pillow`
- **Hardware**:
  - Webcam for data capture and real-time inference.
  - Optional: GPU for faster training (CPU works but is slower).

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy pillow
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/FingerDetectionOpenCV.git
   cd FingerDetectionOpenCV
   ```

2. **Create Dataset Folder**:
   - Create a folder named `captured_dataset` with subfolders `0`, `1`, `2`, `3`, `4`, `5` (one for each gesture class):
     ```bash
     mkdir -p captured_dataset/{0,1,2,3,4,5}
     ```

## Data Collection
1. **Capture Images**:
   - Run `capture-images.py` to collect hand gesture images using your webcam:
     ```bash
     python capture-images.py
     ```
   - Instructions:
     - Place your hand in the green ROI box displayed on the webcam feed.
     - Press keys `0`–`5` to save images to the corresponding class folder (e.g., `captured_dataset/0` for gesture 0).
     - Press `p` to preview the last captured image (32x32 grayscale).
     - Press `q` to quit.
   - Aim for 100–500 images per class for balanced training data.
   - Ensure varied lighting, backgrounds, and hand positions for robustness.

2. **Verify Dataset**:
   - Check that `captured_dataset` contains subfolders `0` to `5`, each with images (32x32 grayscale PNGs).
   - Example command to count images per class:
     ```bash
     for dir in captured_dataset/[0-5]; do echo "$dir: $(ls $dir | wc -l) images"; done
     ```

## Training the Model
1. **Run the Training Script**:
   - Open `train_model.ipynb` in a Jupyter Notebook environment (e.g., VS Code, JupyterLab, or Google Colab).
   - Execute all cells to:
     - Split `captured_dataset` into `train` (64%), `validation` (16%), and `test` (20%) sets.
     - Train a `ConvNeuralNetwork` model with data augmentation and early stopping.
     - Save the best model to `model.pth`.
   - Alternatively, convert the notebook to a `.py` file and run:
     ```bash
     python train_model.py
     ```

2. **Training Details**:
   - **Model**: Convolutional Neural Network (CNN) with two convolutional layers and two fully connected layers.
   - **Input**: 32x32 grayscale images.
   - **Classes**: 6 (gestures 0–5).
   - **Optimizer**: Adam (learning rate 0.001).
   - **Loss**: Cross-entropy.
   - **Evaluation**: Validation set for early stopping, test set for final accuracy.
   - Expected test accuracy: >80% with sufficient, diverse data.

3. **Output**:
   - The script prints training loss, validation accuracy, and final test accuracy.
   - The trained model is saved as `model.pth`.

## Real-Time Inference
1. **Run the Inference Script**:
   - Use `pytorch-opencv.py` to perform real-time gesture recognition with your webcam:
     ```bash
     python pytorch-opencv.py
     ```
   - Instructions:
     - Place your hand in the green ROI box.
     - The script displays the predicted gesture (0–5) and confidence score.
     - Green text indicates high confidence (>0.7); orange indicates lower confidence.
     - Press `q` to quit.

2. **Using the Pre-Trained Model**:
   - The included `model.pth` can be used for inference without retraining.
   - Ensure `pytorch-opencv.py` is in the same directory as `model.pth`.

## Notes
- **Dataset Quality**: For best results, collect diverse images (different lighting, backgrounds, hand positions). Ensure each class has a similar number of images.
- **Overfitting**: If test accuracy is much lower than validation accuracy, add more data or increase augmentation (e.g., random flips).
- **Performance**: Training on CPU is slow; use a GPU if available.
- **Debugging**: If accuracy is low, check dataset balance or inspect misclassifications using a confusion matrix (see `train_model.ipynb` comments for code).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/).
- Inspired by hand gesture recognition tutorials and datasets.