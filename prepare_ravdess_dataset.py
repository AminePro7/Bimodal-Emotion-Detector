import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
from moviepy.editor import VideoFileClip

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

def create_directory_structure():
    """Create necessary directories"""
    dirs = ['test_data/video', 'test_data/audio']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def extract_emotion_from_filename(filename):
    """Extract emotion ID from RAVDESS filename"""
    # Format: 03-01-01-01-01-01-01.mp4
    # Third number is emotion code
    parts = filename.split('-')
    if len(parts) >= 3:
        return parts[2]
    return None

def process_video_file(video_path, output_video_dir, output_audio_dir):
    """Process a single video file to extract frame and audio"""
    try:
        # Get emotion from filename
        filename = os.path.basename(video_path)
        emotion_id = extract_emotion_from_filename(filename)
        if emotion_id not in RAVDESS_EMOTIONS:
            return
        
        # Create base output filename
        base_name = os.path.splitext(filename)[0]
        
        # Extract frame from middle of video
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.set(cv2.CAP_PROP_POS_FRAMES, total_frames//2)
        ret, frame = video.read()
        if ret:
            # Save frame
            frame_path = os.path.join(output_video_dir, f"{emotion_id}_{base_name}.jpg")
            cv2.imwrite(frame_path, frame)
        video.release()
        
        # Extract audio
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join(output_audio_dir, f"{emotion_id}_{base_name}.wav")
        video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video_clip.close()
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")

def prepare_dataset(ravdess_path):
    """Prepare RAVDESS dataset for evaluation"""
    print("Creating directory structure...")
    video_dir, audio_dir = create_directory_structure()
    
    print("Processing RAVDESS videos...")
    # Walk through the RAVDESS directory
    video_files = []
    for root, _, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        process_video_file(video_path, video_dir, audio_dir)
    
    # Print statistics
    print("\nDataset preparation complete!")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Frames saved in: {video_dir}")
    print(f"Audio files saved in: {audio_dir}")

def main():
    # Path to your downloaded RAVDESS dataset
    ravdess_path = "path/to/ravdess/dataset"  # Update this path
    
    if not os.path.exists(ravdess_path):
        print("Please update the ravdess_path variable with the correct path to your downloaded dataset.")
        return
    
    prepare_dataset(ravdess_path)
    print("\nYou can now run evaluate_bimodal_model.py to test the model!")

if __name__ == "__main__":
    main() 