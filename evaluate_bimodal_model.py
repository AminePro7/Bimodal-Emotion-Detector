import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import librosa
from PIL import Image

from bimodal_emotion_detector import BimodalEmotionDetector, late_fusion, DICT_EMO

class BimodalTestDataset:
    def __init__(self, video_dir, audio_dir):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.sample_rate = 16000
        self.samples = []
        
        # Load dataset pairs
        self._load_dataset()
    
    def _load_dataset(self):
        """Load paired video-audio samples with labels"""
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith(('.jpg', '.png')):
                # Get corresponding audio file
                base_name = os.path.splitext(video_file)[0]
                audio_file = os.path.join(self.audio_dir, f"{base_name}.wav")
                
                if os.path.exists(audio_file):
                    # Extract true label from filename (assuming format: emotion_id_*.ext)
                    emotion_id = int(base_name.split('_')[0])
                    
                    self.samples.append({
                        'video_path': os.path.join(self.video_dir, video_file),
                        'audio_path': audio_file,
                        'label': emotion_id
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frame
        video_frame = Image.open(sample['video_path'])
        
        # Load audio
        audio, _ = librosa.load(sample['audio_path'], sr=self.sample_rate)
        
        return video_frame, audio, sample['label']

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_model(model, test_dataset):
    """Evaluate bimodal model on test dataset"""
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Initialize lists to store results
    y_true = []
    y_pred_video = []
    y_pred_audio = []
    y_pred_combined = []
    confidences = {
        'video': [],
        'audio': [],
        'combined': []
    }
    
    print("Starting evaluation...")
    with torch.no_grad():
        for video_frame, audio, label in tqdm(test_dataset):
            # Process video
            static_pred, dynamic_pred, _ = model.process_frame(np.array(video_frame))
            if static_pred is not None and dynamic_pred is not None:
                video_pred = (static_pred + dynamic_pred) / 2
            else:
                video_pred = None
            
            # Process audio
            audio_pred = model.extract_audio_features(audio)
            
            # Late fusion
            if video_pred is not None and audio_pred is not None:
                combined_pred = late_fusion(
                    video_pred,
                    audio_pred,
                    alpha=model.fusion_weights['video']
                )
            else:
                combined_pred = None
            
            # Store true label
            y_true.append(label)
            
            # Store predictions and confidences
            if video_pred is not None:
                video_probs = F.softmax(video_pred, dim=1)
                y_pred_video.append(torch.argmax(video_probs).item())
                confidences['video'].append(torch.max(video_probs).item())
            else:
                y_pred_video.append(-1)
                confidences['video'].append(0.0)
            
            if audio_pred is not None:
                audio_probs = F.softmax(audio_pred, dim=1)
                y_pred_audio.append(torch.argmax(audio_probs).item())
                confidences['audio'].append(torch.max(audio_probs).item())
            else:
                y_pred_audio.append(-1)
                confidences['audio'].append(0.0)
            
            if combined_pred is not None:
                y_pred_combined.append(torch.argmax(combined_pred).item())
                confidences['combined'].append(torch.max(combined_pred).item())
            else:
                y_pred_combined.append(-1)
                confidences['combined'].append(0.0)
    
    # Calculate metrics
    results = {}
    modalities = ['video', 'audio', 'combined']
    predictions = [y_pred_video, y_pred_audio, y_pred_combined]
    
    for modality, preds in zip(modalities, predictions):
        # Filter out invalid predictions (-1)
        valid_indices = [i for i, p in enumerate(preds) if p != -1]
        if not valid_indices:
            continue
            
        valid_preds = [preds[i] for i in valid_indices]
        valid_true = [y_true[i] for i in valid_indices]
        
        # Calculate metrics
        results[modality] = {
            'accuracy': accuracy_score(valid_true, valid_preds),
            'classification_report': classification_report(
                valid_true, valid_preds,
                target_names=list(DICT_EMO.values()),
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(valid_true, valid_preds),
            'avg_confidence': np.mean(confidences[modality])
        }
        
        # Plot confusion matrix
        plot_confusion_matrix(
            valid_true, valid_preds,
            labels=list(DICT_EMO.values())
        )
        plt.savefig(f'confusion_matrix_{modality}.png')
    
    return results

def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n=== Bimodal Emotion Detection Evaluation ===\n")
    
    for modality, metrics in results.items():
        print(f"\n{modality.upper()} MODALITY RESULTS:")
        print("-" * 40)
        
        # Print accuracy and confidence
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
        
        # Print per-class metrics
        print("\nPer-Class Metrics:")
        class_metrics = metrics['classification_report']
        
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        rows = []
        
        for emotion in DICT_EMO.values():
            if emotion in class_metrics:
                metrics_dict = class_metrics[emotion]
                rows.append([
                    emotion,
                    f"{metrics_dict['precision']:.2%}",
                    f"{metrics_dict['recall']:.2%}",
                    f"{metrics_dict['f1-score']:.2%}",
                    f"{metrics_dict['support']}"
                ])
        
        # Print as table
        print("\n" + pd.DataFrame(rows, columns=headers).to_string())
        print("\n" + "=" * 40)

def main():
    # Initialize model
    model = BimodalEmotionDetector()
    
    # Create test dataset
    test_dataset = BimodalTestDataset(
        video_dir='test_data/video',
        audio_dir='test_data/audio'
    )
    
    # Evaluate model
    results = evaluate_model(model, test_dataset)
    
    # Print results
    print_evaluation_results(results)
    
    print("\nEvaluation complete! Confusion matrices saved as PNG files.")

if __name__ == "__main__":
    main() 