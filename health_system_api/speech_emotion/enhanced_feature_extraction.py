import os
import numpy as np
import pandas as pd
import librosa
import glob
from tqdm import tqdm
import random
from scipy.signal import resample
import soundfile as sf

# Define emotion label mapping
emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Define gender label mapping
gender_labels = {
    'odd': 'male',
    'even': 'female'
}

# Define combined label mapping (consistent with the user-provided ONNX code)
combined_labels = {
    'female_angry': 0,
    'female_calm': 1,
    'female_fearful': 2,
    'female_happy': 3,
    'female_sad': 4,
    'male_angry': 5,
    'male_calm': 6,
    'male_fearful': 7,
    'male_happy': 8,
    'male_sad': 9
}

# Data augmentation functions
def augment_audio(signal, sr, augmentation_type=None, noise_factor=0.05, stretch_factor=0.1, pitch_factor=2):
    """
    Augment audio signal
    
    Parameters:
        signal: audio signal
        sr: sampling rate
        augmentation_type: augmentation type (noise, stretch, pitch, None)
        noise_factor: noise factor
        stretch_factor: stretch factor
        pitch_factor: pitch factor
        
    Returns:
        augmented_signal: augmented audio signal
    """
    if augmentation_type is None or len(signal) == 0:
        return signal
    
    augmented_signal = np.copy(signal)
    
    # Add noise
    if augmentation_type == 'noise':
        noise = np.random.randn(len(signal))
        augmented_signal = signal + noise_factor * noise
    
    # Time stretching
    elif augmentation_type == 'stretch':
        stretch_rate = 1 + (random.random() - 0.5) * stretch_factor
        augmented_signal = librosa.effects.time_stretch(signal, rate=stretch_rate)
    
    # Pitch shifting
    elif augmentation_type == 'pitch':
        augmented_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_factor * (random.random() - 0.5))
    
    # Ensure consistent signal length
    if len(augmented_signal) > len(signal):
        augmented_signal = augmented_signal[:len(signal)]
    elif len(augmented_signal) < len(signal):
        augmented_signal = np.pad(augmented_signal, (0, len(signal) - len(augmented_signal)), 'constant')
    
    return augmented_signal

# Extract enhanced feature set
def extract_enhanced_features(file_path, n_mfcc=40, expected_frames=216):
    """
    Extract enhanced feature set from audio file
    
    Parameters:
        file_path: audio file path
        n_mfcc: number of MFCC coefficients
        expected_frames: expected number of frames
        
    Returns:
        features: extracted features
    """
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=44100, duration=3)
        
        # Extract multiple features
        # 1. MFCC features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Chroma features
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        
        # 3. Mel spectrogram
        mel = librosa.feature.melspectrogram(y=signal, sr=sr)
        
        # 4. Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        
        # 5. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        
        # 6. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        
        # 7. Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=signal)
        
        # 8. Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
        
        # Adjust frame count for all features
        def resize_feature(feature, target_frames):
            if feature.shape[1] < target_frames:
                pad_width = target_frames - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :target_frames]
            return feature
        
        # Adjust frame count for all features
        mfccs = resize_feature(mfccs, expected_frames)
        chroma = resize_feature(chroma, expected_frames)
        mel = resize_feature(mel, expected_frames)
        spectral_centroid = resize_feature(spectral_centroid, expected_frames)
        spectral_bandwidth = resize_feature(spectral_bandwidth, expected_frames)
        spectral_contrast = resize_feature(spectral_contrast, expected_frames)
        spectral_flatness = resize_feature(spectral_flatness, expected_frames)
        zero_crossing_rate = resize_feature(zero_crossing_rate, expected_frames)
        
        # Calculate statistics
        def compute_stats(feature):
            mean = np.mean(feature, axis=1)
            std = np.std(feature, axis=1)
            max_val = np.max(feature, axis=1)
            min_val = np.min(feature, axis=1)
            return np.concatenate([mean, std, max_val, min_val])
        
        # Calculate statistics for all features
        mfccs_stats = compute_stats(mfccs)
        chroma_stats = compute_stats(chroma)
        mel_stats = compute_stats(mel)
        spectral_centroid_stats = compute_stats(spectral_centroid)
        spectral_bandwidth_stats = compute_stats(spectral_bandwidth)
        spectral_contrast_stats = compute_stats(spectral_contrast)
        spectral_flatness_stats = compute_stats(spectral_flatness)
        zero_crossing_rate_stats = compute_stats(zero_crossing_rate)
        
        # Combine all features
        features = np.concatenate([
            mfccs_stats,
            chroma_stats,
            mel_stats,
            spectral_centroid_stats,
            spectral_bandwidth_stats,
            spectral_contrast_stats,
            spectral_flatness_stats,
            zero_crossing_rate_stats
        ])
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Parse filename to get labels
def parse_filename(filename):
    """
    Parse RAVDESS dataset filenames to extract emotion and gender information
    
    Parameters:
        filename: file name
        
    Returns:
        emotion: emotion label
        gender: gender label
        combined_label: combined label index
    """
    # Filename format: 03-01-05-01-01-01-01.wav
    # modality-channel-emotion-intensity-statement-repetition-actor
    parts = os.path.basename(filename).split('.')[0].split('-')
    
    emotion_code = parts[2]
    actor_code = parts[6]
    
    # Get emotion label
    emotion = emotion_labels.get(emotion_code, 'unknown')
    
    # Get gender label (odd actor IDs are male, even are female)
    gender = 'male' if int(actor_code) % 2 == 1 else 'female'
    
    # Combined label
    combined_label_key = f"{gender}_{emotion}"
    
    # Only keep the 10 labels we care about (5 emotions x 2 genders)
    if combined_label_key in combined_labels:
        return emotion, gender, combined_labels[combined_label_key]
    else:
        return None, None, None

# Main function: process dataset and extract features
def process_dataset_enhanced(data_path, output_path, n_mfcc=40, expected_frames=216, augment=True):
    """
    Process RAVDESS dataset, extract enhanced features and save
    
    Parameters:
        data_path: dataset path
        output_path: output path
        n_mfcc: number of MFCC coefficients
        expected_frames: expected number of frames
        augment: whether to perform data augmentation
    """
    # Get all audio files
    audio_files = []
    for actor_dir in glob.glob(os.path.join(data_path, "Actor_*")):
        audio_files.extend(glob.glob(os.path.join(actor_dir, "*.wav")))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create feature and label lists
    features = []
    labels = []
    
    # Extract features
    for file_path in tqdm(audio_files, desc="Extracting features"):
        emotion, gender, combined_label = parse_filename(file_path)
        
        # Skip files not in our 10 labels of interest
        if combined_label is None:
            continue
        
        # Extract original features
        feature = extract_enhanced_features(file_path, n_mfcc=n_mfcc, expected_frames=expected_frames)
        
        if feature is not None:
            features.append(feature)
            labels.append(combined_label)
            
            # Data augmentation
            if augment:
                # Load audio file
                signal, sr = librosa.load(file_path, sr=44100, duration=3)
                
                # Apply different types of augmentation
                for aug_type in ['noise', 'stretch', 'pitch']:
                    augmented_signal = augment_audio(signal, sr, augmentation_type=aug_type)
                    
                    # Extract features from augmented signal
                    augmented_feature = extract_enhanced_features(file_path, n_mfcc=n_mfcc, expected_frames=expected_frames)
                    
                    if augmented_feature is not None:
                        features.append(augmented_feature)
                        labels.append(combined_label)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Save features and labels
    np.save(os.path.join(output_path, "enhanced_features.npy"), features)
    np.save(os.path.join(output_path, "enhanced_labels.npy"), labels)
    
    print(f"Enhanced features and labels saved to {output_path}")
    
    return features, labels

if __name__ == "__main__":
    # Dataset path
    data_path = "/home/ubuntu/speech_emotion_analyzer/data/extracted"
    
    # Output path
    output_path = "/home/ubuntu/speech_emotion_analyzer/data/processed"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process dataset
    features, labels = process_dataset_enhanced(data_path, output_path, n_mfcc=40, expected_frames=216, augment=True)