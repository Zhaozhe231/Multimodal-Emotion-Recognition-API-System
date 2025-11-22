import os
import numpy as np
import onnxruntime as ort
import librosa

# Define emotion labels
emotion_labels = {
    0: "female_angry",
    1: "female_calm",
    2: "female_fearful",
    3: "female_happy",
    4: "female_sad",
    5: "male_angry",
    6: "male_calm",
    7: "male_fearful",
    8: "male_happy",
    9: "male_sad"
}

# Extract enhanced features from an audio file.
# Parameters:
#   file_path: Path to the audio file.
#   n_mfcc: Number of MFCC coefficients.
#   expected_frames: Expected number of frames.
# Returns:
#   features: Extracted features.
def extract_enhanced_features(file_path, n_mfcc=40, expected_frames=216):
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=44100, duration=3)
        
        # Extract various features
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
        
        # Adjust the number of frames for each feature
        def resize_feature(feature, target_frames):
            if feature.shape[1] < target_frames:
                pad_width = target_frames - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :target_frames]
            return feature
        
        # Resize all features to have the same number of frames
        mfccs = resize_feature(mfccs, expected_frames)
        chroma = resize_feature(chroma, expected_frames)
        mel = resize_feature(mel, expected_frames)
        spectral_centroid = resize_feature(spectral_centroid, expected_frames)
        spectral_bandwidth = resize_feature(spectral_bandwidth, expected_frames)
        spectral_contrast = resize_feature(spectral_contrast, expected_frames)
        spectral_flatness = resize_feature(spectral_flatness, expected_frames)
        zero_crossing_rate = resize_feature(zero_crossing_rate, expected_frames)
        
        # Compute statistical measures for a feature
        def compute_stats(feature):
            mean = np.mean(feature, axis=1)
            std = np.std(feature, axis=1)
            max_val = np.max(feature, axis=1)
            min_val = np.min(feature, axis=1)
            return np.concatenate([mean, std, max_val, min_val])
        
        # Calculate statistics for each feature
        mfccs_stats = compute_stats(mfccs)
        chroma_stats = compute_stats(chroma)
        mel_stats = compute_stats(mel)
        spectral_centroid_stats = compute_stats(spectral_centroid)
        spectral_bandwidth_stats = compute_stats(spectral_bandwidth)
        spectral_contrast_stats = compute_stats(spectral_contrast)
        spectral_flatness_stats = compute_stats(spectral_flatness)
        zero_crossing_rate_stats = compute_stats(zero_crossing_rate)
        
        # Concatenate all features
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

# EmotionRecognizer class using an ONNX model for inference.
# Methods:
#   __init__(model_path): Initialize the recognizer with a given model.
#   predict(features): Predict emotion using extracted features.
#   predict_file(file_path): Predict the emotion for an audio file.
class EmotionRecognizer:
    def __init__(self, model_path):
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        # Retrieve model input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Model loaded, input name: {self.input_name}, output name: {self.output_name}")
    
    def predict(self, features):
        # Ensure the feature shape is correct
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        # Run the model
        results = self.session.run([self.output_name], {self.input_name: features})
        # Retrieve prediction probabilities
        probabilities = results[0][0]
        # Determine the predicted emotion index
        emotion_idx = np.argmax(probabilities)
        emotion = emotion_labels[emotion_idx]
        
        return emotion, probabilities
    
    def predict_file(self, file_path):
        # Extract features from the audio file
        features = extract_enhanced_features(file_path)
        if features is not None:
            return self.predict(features)
        else:
            return None, None

# Main function to test the emotion recognizer.
# Steps:
#   1. Define model and test audio file paths.
#   2. Create an instance of EmotionRecognizer.
#   3. Predict the emotion for the test audio file.
#   4. Print the predicted emotion and probabilities.
def main():
    model_path = "improved_model.onnx"  # Path to the model
    test_file = "3.wav"                 # Test audio file path，file free to modify
    
    recognizer = EmotionRecognizer(model_path)
    
    emotion, probabilities = recognizer.predict_file(test_file)
    
    if emotion is not None:
        print(f"Predicted emotion: {emotion}")
        print("Probabilities for each emotion:")
        for i, prob in enumerate(probabilities):
            print(f"{emotion_labels[i]}: {prob:.2f}")
    else:
        print("Prediction failed")

if __name__ == "__main__":
    main()
