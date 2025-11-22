import os
import numpy as np
import onnxruntime as ort
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import glob
from tqdm import tqdm

# emotion labels
emotion_labels = {
    0: "female_angry", 1: "female_calm", 2: "female_fearful", 
    3: "female_happy", 4: "female_sad", 5: "male_angry", 
    6: "male_calm", 7: "male_fearful", 8: "male_happy", 9: "male_sad"
}

# Parse filename to get labels
# Parse RAVDESS dataset filenames to extract emotion and gender information
def parse_filename(filename):
    # Filename format: 03-01-05-01-01-01-01.wav
    # modality-channel-emotion-intensity-statement-repetition-actor
    parts = os.path.basename(filename).split('.')[0].split('-')
    
    emotion_code = parts[2]
    actor_code = parts[6]
    
    # Emotion label mapping
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Get emotion label
    emotion = emotion_map.get(emotion_code, 'unknown')
    
    # Get gender label (odd actor IDs are male, even are female)
    gender = 'male' if int(actor_code) % 2 == 1 else 'female'
    
    # Combined label
    combined_label_key = f"{gender}_{emotion}"
    
    # Combined label mapping
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
    
    # Only keep the 10 labels we care about (5 emotions x 2 genders)
    if combined_label_key in combined_labels:
        return emotion, gender, combined_labels[combined_label_key]
    else:
        return None, None, None

# Extract enhanced features
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

# ONNX model inference
class EmotionRecognizer:
    def __init__(self, model_path):
        """
        Initialize emotion recognizer
        
        Parameters:
            model_path: ONNX model path
        """
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded, input name: {self.input_name}, output name: {self.output_name}")
    
    def predict(self, features):
        """
        Make predictions using ONNX model
        
        Parameters:
            features: extracted features
            
        Returns:
            emotion_idx: predicted emotion index
            probabilities: probabilities for each emotion
        """
        # Ensure feature shape is correct
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Make prediction
        results = self.session.run([self.output_name], {self.input_name: features})
        
        # Get prediction results
        probabilities = results[0][0]
        
        # Get predicted emotion index
        emotion_idx = np.argmax(probabilities)
        
        return emotion_idx, probabilities
    
    def predict_file(self, file_path):
        """
        Predict emotion from audio file
        
        Parameters:
            file_path: audio file path
            
        Returns:
            emotion_idx: predicted emotion index
            probabilities: probabilities for each emotion
        """
        # Extract features
        features = extract_enhanced_features(file_path)
        
        if features is not None:
            # Make prediction
            return self.predict(features)
        else:
            return None, None
        
    #Parameters: data_path: dataset path    
    #Returns: accuracy: accuracy score.y_true: true labels. y_pred: predicted labels
    def evaluate_dataset(self, data_path):
        # Get all audio files
        audio_files = []
        for actor_dir in glob.glob(os.path.join(data_path, "Actor_*")):
            audio_files.extend(glob.glob(os.path.join(actor_dir, "*.wav")))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Create true and predicted label lists
        y_true = []
        y_pred = []
        
        # Evaluate each audio file
        for file_path in tqdm(audio_files, desc="Evaluating dataset"):
            # Get true label
            _, _, true_label = parse_filename(file_path)
            
            # Skip files not in our 10 labels of interest
            if true_label is None:
                continue
            
            # Predict emotion
            pred_label, _ = self.predict_file(file_path)
            
            if pred_label is not None:
                y_true.append(true_label)
                y_pred.append(pred_label)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        return accuracy, y_true, y_pred
    
    def visualize_confusion_matrix(self, y_true, y_pred):
        """
        Visualize confusion matrix
        
        Parameters:
            y_true: true labels
            y_pred: predicted labels
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names
        class_names = [emotion_labels[i] for i in range(len(emotion_labels))]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        
        # Set title and labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save figure
        plt.savefig("/home/ubuntu/speech_emotion_analyzer/results/onnx_confusion_matrix.png")
        plt.close()
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate classification report
        
        Parameters:
            y_true: true labels
            y_pred: predicted labels
            
        Returns:
            report: classification report
        """
        # Get class names
        class_names = [emotion_labels[i] for i in range(len(emotion_labels))]
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        return report

# Main function
def main():
    # Model path
    model_path = "/home/ubuntu/speech_emotion_analyzer/models/improved_model.onnx"
    
    # Dataset path
    data_path = "/home/ubuntu/speech_emotion_analyzer/data/extracted"
    
    # Create emotion recognizer
    recognizer = EmotionRecognizer(model_path)
    
    # Evaluate dataset
    accuracy, y_true, y_pred = recognizer.evaluate_dataset(data_path)
    
    print(f"ONNX model accuracy on test set: {accuracy*100:.2f}%")
    
    # Visualize confusion matrix
    recognizer.visualize_confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = recognizer.generate_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save evaluation results
    with open("/home/ubuntu/speech_emotion_analyzer/results/onnx_evaluation_results.txt", "w") as f:
        f.write(f"ONNX model accuracy on test set: {accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Evaluation results saved to /home/ubuntu/speech_emotion_analyzer/results/onnx_evaluation_results.txt")

if __name__ == "__main__":
    main()