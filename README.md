# 🎭 Bimodal Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff6f00.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-AminePro7-181717.svg?logo=github)](https://github.com/AminePro7)

A real-time bimodal emotion detection system that combines facial expression recognition and speech emotion recognition for robust emotion analysis.

![Bimodal Emotion Detection Demo](https://via.placeholder.com/800x400?text=Bimodal+Emotion+Detection+Demo)

## 🚀 Features

- **🔄 Real-time Processing**: Simultaneous processing of video and audio inputs
- **😀 Facial Emotion Recognition**:
  - Static analysis using ResNet50 trained on AffectNet
  - Dynamic analysis using LSTM trained on various datasets (IEMOCAP, CREMA-D, RAVDESS, etc.)
  - Face detection using MediaPipe Face Mesh
- **🔊 Speech Emotion Recognition**:
  - Audio processing using mel-spectrograms
  - Pre-trained model on RAVDESS dataset
- **🎯 Emotion Categories**:
  - Neutral
  - Happy
  - Sad
  - Surprise
  - Fear
  - Disgust
  - Anger

## 🖥️ System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for better performance)
- Webcam
- Microphone

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/AminePro7/Bimodal-Emotion-Detector.git
cd Bimodal-Emotion-Detector

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if not included in the repository)
# Instructions for downloading models would go here
```

## 📁 Project Structure

- `bimodal_emotion_detector.py`: Main implementation of the bimodal system
- `run_webcam.py`: Implementation of facial emotion recognition
- `train_audio_model.py`: Training script for audio emotion model
- `evaluate_bimodal_model.py`: Evaluation script for the bimodal system
- `prepare_ravdess_dataset.py`: Script for preparing the RAVDESS dataset
- Pre-trained models:
  - `FER_static_ResNet50_AffectNet.pt`: Static facial emotion recognition
  - `FER_dinamic_LSTM_*.pt`: Dynamic facial emotion recognition models
  - `audio-ravdness-model.h5`: Speech emotion recognition

## 🏗️ Model Architecture

### 👁️ Visual Stream
1. **Static Analysis**:
   - ResNet50 backbone
   - Trained on AffectNet dataset
   - Real-time face detection and emotion classification

2. **Dynamic Analysis**:
   - LSTM network for temporal features
   - Processes sequences of facial features
   - Captures temporal emotion patterns

### 👂 Audio Stream
- Mel-spectrogram feature extraction
- Input shape: (352, 15) time steps and features
- Pre-trained on RAVDESS dataset
- Real-time audio processing with sliding window

### 🔄 Fusion Strategy
- Late fusion combining predictions from both modalities
- Weighted averaging of predictions
- Real-time synchronization of audio and video streams

## 🚀 Usage

1. Run the bimodal emotion detector:
```bash
python bimodal_emotion_detector.py
```

2. The system will display:
   - Video feed with detected face and emotion
   - Real-time emotion predictions:
     - Video-based (green)
     - Audio-based (red)
     - Combined (purple)
   - Press 'q' to quit

## ⚙️ Performance Optimizations

- Reduced frame resolution and processing rate
- Efficient memory management
- Minimal buffer sizes for audio and video
- CPU thread limiting for stable performance
- Aggressive garbage collection

## 🧠 Training Custom Models

### Audio Model Training
```bash
python train_audio_model.py
```
- Requires RAVDESS dataset
- Supports custom dataset integration
- Automatic train/validation split
- Model checkpointing

### Preparing RAVDESS Dataset
```bash
python prepare_ravdess_dataset.py
```

## 🗺️ Emotion Mapping

### Video Emotions
- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Surprise
- 4: Fear
- 5: Disgust
- 6: Anger

### Audio Emotions (RAVDESS)
- Neutral
- Calm (mapped to Neutral)
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise

## 📊 Evaluation

To evaluate the bimodal model:
```bash
python evaluate_bimodal_model.py
```

## 🚧 Limitations and Future Work

1. **Current Limitations**:
   - High memory usage during initialization
   - Fixed emotion categories
   - Limited to frontal face views
   - Requires good lighting conditions

2. **Potential Improvements**:
   - Model compression for better performance
   - Additional emotion categories
   - Multi-face support
   - Cross-cultural emotion recognition
   - More sophisticated fusion strategies

## 📝 Citation

If you use this code for your research, please cite:

```
@software{bimodal_emotion_detector,
  author = {AminePro7},
  title = {Bimodal Emotion Detection System},
  url = {https://github.com/AminePro7/Bimodal-Emotion-Detector},
  year = {2024},
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created by [AminePro7](https://github.com/AminePro7)

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for face detection
- [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) for deep learning frameworks
- [RAVDESS](https://zenodo.org/record/1188976) dataset for audio emotion recognition
- [AffectNet](http://mohammadmahoor.com/affectnet/) for facial emotion recognition

