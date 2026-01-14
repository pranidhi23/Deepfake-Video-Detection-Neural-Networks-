# Deepfake Detection of Manipulated Videos

## Project Overview
This project focuses on detecting deepfake (manipulated) videos using deep learning techniques.
A hybrid CNN-RNN model is implemented to capture both spatial and temporal features from video frames.

The goal is to identify whether a video is real or manipulated with high accuracy.

## Dataset
- FaceForensics++ dataset
- Contains real and manipulated videos generated using multiple deepfake techniques

## Model Architecture
- Convolutional Neural Network (CNN) for spatial feature extraction from video frames
- Recurrent Neural Network (RNN) to capture temporal dependencies across frames
- Hybrid CNN-RNN architecture for improved detection performance

## Tools & Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Methodology
1. Video preprocessing and frame extraction
2. Feature extraction using CNN
3. Temporal sequence modeling using RNN
4. Classification of videos as real or fake

## Key Outcomes
- Successfully detected manipulated videos using deep learning techniques
- Demonstrated the effectiveness of combining CNN and RNN for video-based classification
- Improved understanding of video forensics and AI-based media authentication

## Applications
- Media authenticity verification
- Social media content moderation
- Cybersecurity and digital forensics

## Future Improvements
- Train on larger and more diverse datasets
- Improve model accuracy using attention mechanisms
- Deploy the model as a web-based application

## Author
K K Pranidhi  
B.E Computer Science & Engineering(Design)
