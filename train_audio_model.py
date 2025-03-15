import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import tqdm
from bimodal_emotion_detector import AudioEmotionModel

# RAVDESS emotion mapping
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

# Map RAVDESS emotions to our model's emotions (0-6)
EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'fear': 4,
    'disgust': 5,
    'angry': 6
}

class RAVDESSDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.file_list = []
        self.labels = []
        
        # Walk through the dataset directory
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    # Parse emotion from filename
                    # Format: 03-01-01-01-01-01-01.wav
                    # Third number is emotion code
                    emotion_code = file.split('-')[2]
                    emotion = RAVDESS_EMOTIONS[emotion_code]
                    
                    # Only use emotions that map to our model
                    if emotion in EMOTION_MAP:
                        self.file_list.append(os.path.join(root, file))
                        self.labels.append(EMOTION_MAP[emotion])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        audio, _ = librosa.load(audio_file, sr=self.sample_rate, duration=3)
        
        # Ensure consistent length by padding or truncating
        target_length = 3 * self.sample_rate  # 3 seconds
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Pad or truncate to match expected input size (13, 128)
        if mfccs.shape[1] < 128:
            mfccs = np.pad(mfccs, ((0, 0), (0, 128 - mfccs.shape[1])))
        else:
            mfccs = mfccs[:, :128]
        
        # Convert to tensor
        mfccs_tensor = torch.FloatTensor(mfccs).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.LongTensor([label])
        
        return mfccs_tensor, label_tensor

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{train_loss/train_total:.3f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss/train_total:.3f}, '
              f'Train Acc: {100.*train_correct/train_total:.2f}%, '
              f'Val Loss: {val_loss:.3f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_audio_model.pt')
        
        scheduler.step(val_loss)
    
    return best_model_state

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    dataset = RAVDESSDataset('RAVDESS')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = AudioEmotionModel().to(device)
    
    # Train model
    best_model_state = train_model(model, train_loader, val_loader, device=device)
    
    # Save final model
    torch.save(best_model_state, 'audio_model_final.pt')
    print('Training completed. Model saved as audio_model_final.pt')

if __name__ == '__main__':
    main() 