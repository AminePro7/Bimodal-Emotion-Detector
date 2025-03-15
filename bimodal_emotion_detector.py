import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sounddevice as sd
import librosa
import threading
import queue
import time
from typing import Dict, Tuple, List
import tensorflow as tf
import gc

# Import the existing FER models
from run_webcam import ResNet50, LSTMPyTorch, pth_processing, DICT_EMO, display_EMO_PRED, display_FPS

# Audio model emotion mapping (if different from video model)
AUDIO_EMOTION_MAP = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fear',
    6: 'disgust',
    7: 'surprise'
}

def map_audio_to_video_emotion(audio_emotion_idx):
    """Map audio model emotion index to video model emotion index"""
    audio_emotion = AUDIO_EMOTION_MAP[audio_emotion_idx]
    # Map to closest matching video emotion
    emotion_map = {
        'neutral': 0,  # Neutral
        'calm': 0,     # Map calm to neutral
        'happy': 1,    # Happy
        'sad': 2,      # Sad
        'surprise': 3, # Surprise
        'fear': 4,     # Fear
        'disgust': 5,  # Disgust
        'angry': 6     # Angry
    }
    return emotion_map.get(audio_emotion, 0)  # Default to neutral if unknown

def draw_text_with_background(img, text, position, font_scale=1, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, 
                            text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=10):
    """Helper function to draw text with background"""
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    # Draw background rectangle with alpha blending for transparency
    overlay = img.copy()
    cv2.rectangle(overlay, (x - padding, y - text_h - padding), 
                 (x + text_w + padding, y + padding), bg_color, -1)
    # Apply transparency
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    return img

def draw_predictions(frame, video_pred=None, audio_pred=None, combined_pred=None):
    """Helper function to draw all predictions with consistent styling"""
    y_spacing = 35  # Space between lines
    x_pos = 10
    y_start = 40
    
    # Video prediction (Green)
    if video_pred is not None:
        video_idx = torch.argmax(video_pred).item()
        video_emotion = DICT_EMO[video_idx]
        video_conf = video_pred[0][video_idx].item()
        video_text = f"Video: {video_emotion} ({video_conf:.1%})"
    else:
        video_text = "Video: No face detected"
    frame = draw_text_with_background(frame, video_text, 
                                    (x_pos, y_start), 0.7, 
                                    bg_color=(0, 100, 0))
    
    # Audio prediction (Red)
    if audio_pred is not None:
        audio_idx = torch.argmax(audio_pred).item()
        audio_emotion = DICT_EMO[audio_idx]
        audio_conf = audio_pred[0][audio_idx].item()
        audio_text = f"Audio: {audio_emotion} ({audio_conf:.1%})"
    else:
        audio_text = "Audio: No audio detected"
    frame = draw_text_with_background(frame, audio_text, 
                                    (x_pos, y_start + y_spacing), 0.7, 
                                    bg_color=(100, 0, 0))
    
    # Combined prediction (Blue)
    if combined_pred is not None:
        combined_idx = torch.argmax(combined_pred).item()
        combined_emotion = DICT_EMO[combined_idx]
        combined_conf = combined_pred[0][combined_idx].item()
        combined_text = f"Combined: {combined_emotion} ({combined_conf:.1%})"
    else:
        if video_pred is not None or audio_pred is not None:
            combined_text = "Combined: Waiting for both inputs"
        else:
            combined_text = "Combined: No inputs detected"
    frame = draw_text_with_background(frame, combined_text, 
                                    (x_pos, y_start + 2 * y_spacing), 0.7, 
                                    bg_color=(100, 0, 100))
    
    return frame

def late_fusion(video_pred, audio_pred, alpha=0.7):
    """
    Implement late fusion strategy with weighted combination
    Args:
        video_pred: Video modality predictions (tensor)
        audio_pred: Audio modality predictions (tensor)
        alpha: Weight for video modality (1-alpha for audio)
    Returns:
        Combined prediction using late fusion
    """
    if video_pred is None and audio_pred is None:
        return None
    elif video_pred is None:
        return audio_pred
    elif audio_pred is None:
        return video_pred
    
    # Apply softmax to get probability distributions
    video_probs = F.softmax(video_pred, dim=1)
    audio_probs = F.softmax(audio_pred, dim=1)
    
    # Weighted combination
    combined_pred = alpha * video_probs + (1 - alpha) * audio_probs
    
    return combined_pred

class AudioProcessor:
    def __init__(self, model_path='audio-ravdness-model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.sample_rate = 16000
        self.chunk_size = 16000  # 1 second of audio
        
    def preprocess_audio(self, audio_data):
        """Preprocess audio data for the model with memory efficiency"""
        try:
            # Convert to float32 and flatten
            audio_data = audio_data.astype(np.float32).flatten()
            
            # Trim or pad to chunk size
            if len(audio_data) > self.chunk_size:
                audio_data = audio_data[:self.chunk_size]
            else:
                audio_data = np.pad(audio_data, (0, self.chunk_size - len(audio_data)))
            
            # Extract features
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=15,  # Match the expected feature dimension
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window='hann',
                center=True,
                pad_mode='reflect',
                power=2.0
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            
            # Resize to match expected dimensions (352, 15)
            target_length = 352
            if mel_spec_db.shape[1] != target_length:
                mel_spec_db = np.array([np.interp(
                    np.linspace(0, 1, target_length),
                    np.linspace(0, 1, mel_spec_db.shape[1]),
                    row
                ) for row in mel_spec_db])
            
            # Transpose to match expected shape (time_steps, features)
            mel_spec_db = mel_spec_db.T
            
            # Add batch dimension
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error in preprocess_audio: {str(e)}")
            return None
        
    def predict(self, audio_data):
        """Predict emotion from audio data and convert to PyTorch format"""
        try:
            # Preprocess audio
            features = self.preprocess_audio(audio_data)
            if features is None:
                return None
            
            # Get prediction from TensorFlow model
            pred = self.model.predict(features, verbose=0)
            
            # Convert prediction to match video model format
            pred_idx = np.argmax(pred[0])
            video_emotion_idx = map_audio_to_video_emotion(pred_idx)
            
            # Create PyTorch-style output
            output = np.zeros((1, 7))  # 7 emotions to match video model
            output[0, video_emotion_idx] = 1.0
            return torch.FloatTensor(output)
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return None

class BimodalEmotionDetector:
    def __init__(self):
        try:
            print("Loading facial emotion models...")
            # Initialize models with reduced memory footprint
            self.device = torch.device('cpu')
            torch.set_num_threads(2)  # Limit CPU threads
            
            # Load models with reduced memory
            self.fer_static = ResNet50(num_classes=7)
            self.fer_static.load_state_dict(torch.load('FER_static_ResNet50_AffectNet.pt', map_location=self.device))
            self.fer_static.eval()
            
            self.fer_dynamic = LSTMPyTorch()
            self.fer_dynamic.load_state_dict(torch.load('FER_dinamic_LSTM_Aff-Wild2.pt', map_location=self.device))
            self.fer_dynamic.eval()
            
            # Initialize audio processor
            print("Loading audio model...")
            self.audio_processor = AudioProcessor('audio-ravdness-model.h5')
            
            # MediaPipe face detection with minimal resource usage
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False)
            
            # Minimal buffer sizes
            self.sample_rate = 16000
            self.audio_window = 1.0  # 1 second window to match audio processing
            self.audio_buffer = queue.Queue(maxsize=1)  # Only keep latest audio
            self.lstm_features = []
            self.max_lstm_buffer = 3
            
            # Enable aggressive garbage collection
            gc.enable()
            gc.set_threshold(100, 5, 5)
            
            # Add fusion parameters
            self.fusion_weights = {
                'video': 0.7,  # Video modality weight
                'audio': 0.3   # Audio modality weight
            }
            
            # Add metrics tracking
            self.metrics = {
                'video_confidences': [],
                'audio_confidences': [],
                'combined_confidences': [],
                'modality_agreement': [],
                'fusion_weights_history': [],
                'predictions_count': {emotion: 0 for emotion in DICT_EMO.values()}
            }
            self.metrics_window = 30  # Track metrics for last 30 frames
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    
    def process_audio(self, indata, frames, time, status):
        """Callback for audio stream processing"""
        try:
            if status:
                print(status)
            # Only keep the latest audio data
            if not self.audio_buffer.full():
                self.audio_buffer.put(indata.copy())
            else:
                try:
                    self.audio_buffer.get_nowait()
                    self.audio_buffer.put(indata.copy())
                except queue.Empty:
                    pass
        except Exception as e:
            print(f"Error in audio callback: {str(e)}")
            
    def extract_audio_features(self, audio_data):
        """Process audio data using the pre-trained model"""
        return self.audio_processor.predict(audio_data)
        
    def process_frame(self, frame):
        """Process a single video frame with minimal memory usage"""
        try:
            # Aggressive downscaling
            scale_factor = 0.25  # Even smaller frames
            frame_small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_AREA)
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None, None, frame
                
            h, w, _ = frame_small.shape
            fl = results.multi_face_landmarks[0]
            from run_webcam import get_box
            startX, startY, endX, endY = get_box(fl, w, h)
            face_roi = frame_rgb[startY:endX, startX:endX]
            
            if face_roi.size == 0:
                return None, None, frame
                
            # Process face with minimal memory
            face_pil = Image.fromarray(face_roi)
            face_tensor = pth_processing(face_pil)
            
            # Get predictions
            with torch.no_grad():
                features = self.fer_static.extract_features(face_tensor)
                static_pred = self.fer_static(face_tensor)
                
                # Update LSTM buffer
                if len(self.lstm_features) == 0:
                    self.lstm_features = [features.numpy()] * self.max_lstm_buffer
                else:
                    self.lstm_features = self.lstm_features[1:] + [features.numpy()]
                
                # Get dynamic prediction
                lstm_f = torch.from_numpy(np.vstack(self.lstm_features))
                lstm_f = lstm_f.unsqueeze(0)
                dynamic_pred = self.fer_dynamic(lstm_f)
            
            # Scale coordinates back to original size
            startX = int(startX / scale_factor)
            startY = int(startY / scale_factor)
            endX = int(endX / scale_factor)
            endY = int(endY / scale_factor)
            
            # Clear memory
            del frame_small, frame_rgb, face_roi, face_tensor, features, lstm_f
            gc.collect()
            
            return static_pred, dynamic_pred, (startX, startY, endX, endY)
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return None, None, frame
    
    def update_fusion_weights(self, video_weight=0.7):
        """Update fusion weights dynamically"""
        self.fusion_weights['video'] = video_weight
        self.fusion_weights['audio'] = 1.0 - video_weight
    
    def update_metrics(self, video_pred, audio_pred, combined_pred):
        """Update evaluation metrics"""
        try:
            # Get predictions and confidences
            metrics = {}
            
            if video_pred is not None:
                video_probs = F.softmax(video_pred, dim=1)
                video_conf = torch.max(video_probs).item()
                video_emotion = DICT_EMO[torch.argmax(video_probs).item()]
                metrics['video'] = {'emotion': video_emotion, 'confidence': video_conf}
                
                # Update video confidence history
                self.metrics['video_confidences'].append(video_conf)
                if len(self.metrics['video_confidences']) > self.metrics_window:
                    self.metrics['video_confidences'].pop(0)
                
                # Update emotion counts
                self.metrics['predictions_count'][video_emotion] += 1
            
            if audio_pred is not None:
                audio_probs = F.softmax(audio_pred, dim=1)
                audio_conf = torch.max(audio_probs).item()
                audio_emotion = DICT_EMO[torch.argmax(audio_probs).item()]
                metrics['audio'] = {'emotion': audio_emotion, 'confidence': audio_conf}
                
                # Update audio confidence history
                self.metrics['audio_confidences'].append(audio_conf)
                if len(self.metrics['audio_confidences']) > self.metrics_window:
                    self.metrics['audio_confidences'].pop(0)
            
            if combined_pred is not None:
                combined_conf = torch.max(combined_pred).item()
                combined_emotion = DICT_EMO[torch.argmax(combined_pred).item()]
                metrics['combined'] = {'emotion': combined_emotion, 'confidence': combined_conf}
                
                # Update combined confidence history
                self.metrics['combined_confidences'].append(combined_conf)
                if len(self.metrics['combined_confidences']) > self.metrics_window:
                    self.metrics['combined_confidences'].pop(0)
            
            # Calculate modality agreement
            if 'video' in metrics and 'audio' in metrics:
                agreement = 1.0 if metrics['video']['emotion'] == metrics['audio']['emotion'] else 0.0
                self.metrics['modality_agreement'].append(agreement)
                if len(self.metrics['modality_agreement']) > self.metrics_window:
                    self.metrics['modality_agreement'].pop(0)
            
            # Track fusion weights
            self.metrics['fusion_weights_history'].append(self.fusion_weights.copy())
            if len(self.metrics['fusion_weights_history']) > self.metrics_window:
                self.metrics['fusion_weights_history'].pop(0)
            
            # Log metrics every 30 frames
            if len(self.metrics['video_confidences']) == self.metrics_window:
                self.log_metrics()
                
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")

    def log_metrics(self):
        """Log current evaluation metrics"""
        try:
            print("\n=== Emotion Detection Metrics ===")
            
            # Average confidences
            if self.metrics['video_confidences']:
                avg_video_conf = sum(self.metrics['video_confidences']) / len(self.metrics['video_confidences'])
                print(f"Video Confidence (avg): {avg_video_conf:.2%}")
            
            if self.metrics['audio_confidences']:
                avg_audio_conf = sum(self.metrics['audio_confidences']) / len(self.metrics['audio_confidences'])
                print(f"Audio Confidence (avg): {avg_audio_conf:.2%}")
            
            if self.metrics['combined_confidences']:
                avg_combined_conf = sum(self.metrics['combined_confidences']) / len(self.metrics['combined_confidences'])
                print(f"Combined Confidence (avg): {avg_combined_conf:.2%}")
            
            # Modality agreement
            if self.metrics['modality_agreement']:
                agreement_rate = sum(self.metrics['modality_agreement']) / len(self.metrics['modality_agreement'])
                print(f"Modality Agreement Rate: {agreement_rate:.2%}")
            
            # Current fusion weights
            print(f"Current Fusion Weights: Video={self.fusion_weights['video']:.2f}, Audio={self.fusion_weights['audio']:.2f}")
            
            # Emotion distribution
            total_predictions = sum(self.metrics['predictions_count'].values())
            if total_predictions > 0:
                print("\nEmotion Distribution:")
                for emotion, count in self.metrics['predictions_count'].items():
                    percentage = (count / total_predictions) * 100
                    print(f"{emotion}: {percentage:.1f}%")
            
            print("===============================\n")
            
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")

    def run(self):
        """Run with minimal resource usage"""
        try:
            print("Opening webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("Could not open webcam")
            
            # Set very low resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS
            
            print("Starting audio stream...")
            audio_stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.process_audio,
                blocksize=int(self.sample_rate * self.audio_window),
                device=None,
                latency='high'  # Use high latency for stability
            )
            
            print("Starting main loop...")
            frame_count = 0
            with audio_stream:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every third frame
                    frame_count += 1
                    if frame_count % 3 != 0:
                        continue
                    
                    try:
                        # Initialize variables
                        static_pred = None
                        dynamic_pred = None
                        audio_pred = None
                        video_pred = None
                        combined_pred = None
                        bbox = None
                        
                        # Process frame
                        static_pred, dynamic_pred, bbox = self.process_frame(frame)
                        
                        # Process audio
                        try:
                            audio_data = self.audio_buffer.get_nowait()
                            audio_pred = self.extract_audio_features(audio_data)
                        except queue.Empty:
                            audio_pred = None
                        
                        # Update display with late fusion
                        if static_pred is not None and dynamic_pred is not None:
                            # Combine static and dynamic predictions for video
                            video_pred = (static_pred + dynamic_pred) / 2
                            
                            # Apply late fusion if both modalities are available
                            if audio_pred is not None:
                                combined_pred = late_fusion(
                                    video_pred, 
                                    audio_pred, 
                                    alpha=self.fusion_weights['video']
                                )
                            
                            # Confidence-based weight adjustment (optional)
                            if video_pred is not None and audio_pred is not None:
                                video_conf = torch.max(F.softmax(video_pred, dim=1)).item()
                                audio_conf = torch.max(F.softmax(audio_pred, dim=1)).item()
                                
                                # Adjust weights based on confidence
                                if abs(video_conf - audio_conf) > 0.3:  # Significant confidence difference
                                    if video_conf > audio_conf:
                                        self.update_fusion_weights(0.8)  # Increase video weight
                                    else:
                                        self.update_fusion_weights(0.6)  # Decrease video weight
                                else:
                                    self.update_fusion_weights(0.7)  # Reset to default
                        
                        # Update metrics
                        self.update_metrics(video_pred, audio_pred, combined_pred)
                        
                        # Draw predictions
                        frame = draw_predictions(frame, video_pred, audio_pred, combined_pred)
                        if bbox is not None and video_pred is not None:
                            video_idx = torch.argmax(video_pred).item()
                            video_emotion = DICT_EMO[video_idx]
                            video_conf = F.softmax(video_pred, dim=1)[0][video_idx].item()
                            frame = display_EMO_PRED(frame, bbox, f"{video_emotion} {video_conf:.1%}", line_width=2)
                        
                        # Display frame
                        cv2.imshow('Bimodal Emotion Detection', frame)
                        
                    except Exception as e:
                        print(f"Error in main loop: {str(e)}")
                        continue
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
        except Exception as e:
            print(f"Fatal error: {str(e)}")
        
        finally:
            # Print final metrics
            print("\n=== Final Metrics ===")
            self.log_metrics()
            
            # Cleanup
            print("Cleaning up...")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            self.audio_buffer.queue.clear()
            self.lstm_features.clear()
            gc.collect()
            print("Cleanup complete")

if __name__ == "__main__":
    try:
        print("Starting Bimodal Emotion Detector...")
        detector = BimodalEmotionDetector()
        detector.run()
    except Exception as e:
        print(f"Program terminated due to error: {str(e)}")
    finally:
        print("Program ended") 