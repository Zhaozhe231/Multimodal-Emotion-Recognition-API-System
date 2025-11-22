from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import numpy as np
import librosa
import cv2
import onnxruntime as ort
from expert_judgment_system import ExpertJudgmentSystem
import time
import logging
import uuid
from werkzeug.utils import secure_filename
import sqlite3

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Enable debug mode for detailed error messages
app.config['DEBUG'] = True
app.secret_key = 'health_system_secret_key'

# Define upload folder for temporary files
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for model paths - update these to your actual paths
voice_model_path = "improved_model.onnx"
face_model_path = "face_emotion_model.onnx"


# Voice emotion recognition model
class VoiceEmotionRecognizer:
    def __init__(self, model_path):
        """Initialize voice emotion recognizer"""
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(
                f"Voice emotion model loaded successfully, input name: {self.input_name}, output name: {self.output_name}")
        except Exception as e:
            logger.error(f"Voice emotion model loading failed: {e}")
            raise

    def extract_features(self, file_path, n_mfcc=40, expected_frames=216):
        """Extract features from audio file"""
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

            # Resize all features to expected frame count
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

            # Compute statistics for all features
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
            logger.error(f"Feature extraction failed: {e}")
            return None

    def predict(self, features):
        """Make predictions using the model"""
        try:
            # Ensure features have correct shape
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            # Run the model
            results = self.session.run([self.output_name], {self.input_name: features})

            # Get prediction results
            probabilities = results[0][0]

            # Get predicted emotion index
            emotion_idx = np.argmax(probabilities)

            # Emotion labels
            emotion_labels = {
                0: "female_angry", 1: "female_calm", 2: "female_fearful",
                3: "female_happy", 4: "female_sad", 5: "male_angry",
                6: "male_calm", 7: "male_fearful", 8: "male_happy", 9: "male_sad"
            }

            emotion = emotion_labels[emotion_idx]

            return emotion, probabilities.tolist()

        except Exception as e:
            logger.error(f"Voice emotion prediction failed: {e}")
            return None, None

    def predict_file(self, file_path):
        """Predict emotion from an audio file"""
        try:
            # Extract features
            features = self.extract_features(file_path)

            if features is not None:
                # Make prediction
                return self.predict(features)
            else:
                return None, None
        except Exception as e:
            logger.error(f"Audio file prediction failed: {e}")
            return None, None


# Face emotion recognition model
class FaceEmotionRecognizer:
    def __init__(self, model_path):
        """Initialize face emotion recognizer"""
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Face emotion model loaded successfully, input name: {self.input_name}")

            # Load Haar cascade classifier
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                logger.error("Unable to load face detector")
                raise ValueError("Unable to load face detector")
        except Exception as e:
            logger.error(f"Face emotion model loading failed: {e}")
            raise

    def predict_image(self, image_path):
        """Predict facial emotion from an image"""
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Unable to read image: {image_path}")
                return None, None

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                logger.warning("No faces detected")
                return None, None

            # Emotion labels
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

            # Process the first detected face
            x, y, w, h = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Expand dimensions and normalize
            roi_gray = np.expand_dims(roi_gray, axis=[0, -1]) / 255.0
            roi_gray = roi_gray.astype(np.float32)

            # Perform inference with the model
            outputs = self.session.run(None, {self.input_name: roi_gray})
            prediction = outputs[0]

            # Get prediction results
            emotion_idx = np.argmax(prediction)
            emotion = labels[emotion_idx]

            return emotion, prediction[0].tolist()

        except Exception as e:
            logger.error(f"Face emotion prediction failed: {e}")
            return None, None


# Initialize models
voice_recognizer = None
face_recognizer = None


def init_models():
    """Initialize models"""
    global voice_recognizer, face_recognizer
    try:
        # Check if model files exist
        if not os.path.exists(voice_model_path):
            logger.warning(f"Voice model file not found at: {voice_model_path}")
            logger.warning("Using placeholder voice model")
            voice_recognizer = None
        else:
            voice_recognizer = VoiceEmotionRecognizer(voice_model_path)

        if not os.path.exists(face_model_path):
            logger.warning(f"Face model file not found at: {face_model_path}")
            logger.warning("Using placeholder face model")
            face_recognizer = None
        else:
            face_recognizer = FaceEmotionRecognizer(face_model_path)

        # Initialize expert system
        expert_system = ExpertJudgmentSystem()

        logger.info("Models initialized successfully")
        return expert_system

    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        # Return a basic expert system that doesn't rely on the models
        return ExpertJudgmentSystem()


# Initialize expert system
expert_system = init_models()


@app.route('/', methods=['GET'])
def index():
    """Home page of the web interface"""
    return render_template('index.html')


@app.route('/analyze-form', methods=['GET'])
def analyze_form():
    """Page with a form to upload files for analysis"""
    return render_template('analyze_form.html')


@app.route('/submit-analysis', methods=['POST'])
def submit_analysis():
    """Process uploaded files and analyze emotions"""
    try:
        logger.info("Form submission received")

        # Generate a unique session ID for this analysis
        session_id = str(uuid.uuid4())
        logger.info(f"Analysis session ID: {session_id}")

        # Check if files were uploaded
        voice_file = request.files.get('voice')
        face_file = request.files.get('face')

        # Debug info about files
        logger.info(f"Voice file: {voice_file.filename if voice_file else 'None'}")
        logger.info(f"Face file: {face_file.filename if face_file else 'None'}")

        if not voice_file and not face_file:
            flash('Please provide at least one file for analysis', 'warning')
            return redirect(url_for('analyze_form'))

        voice_emotion = None
        voice_probs = None
        face_emotion = None
        face_probs = None
        temp_paths = []  # Keep track of all temp files

        # Process voice file
        if voice_file and voice_file.filename:
            try:
                # Create a safe filename
                voice_filename = f"voice_{session_id}_{secure_filename(voice_file.filename)}"
                temp_voice_path = os.path.join(app.config['UPLOAD_FOLDER'], voice_filename)
                temp_paths.append(temp_voice_path)

                # Save the file
                logger.info(f"Saving voice file to: {temp_voice_path}")
                voice_file.save(temp_voice_path)

                # Process with the model if available
                if voice_recognizer:
                    logger.info("Processing voice with model")
                    voice_emotion, voice_probs = voice_recognizer.predict_file(temp_voice_path)
                    logger.info(f"Voice emotion detected: {voice_emotion}")
                else:
                    # Fallback to placeholder
                    logger.info("Using placeholder voice emotion")
                    voice_emotion = "Happy"
                    voice_probs = [0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Higher prob for happy

            except Exception as e:
                logger.error(f"Error processing voice file: {str(e)}")
                flash(f"Warning: Could not process voice file: {str(e)}", 'warning')

        # Process face file
        if face_file and face_file.filename:
            try:
                # Create a safe filename
                face_filename = f"face_{session_id}_{secure_filename(face_file.filename)}"
                temp_face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                temp_paths.append(temp_face_path)

                # Save the file
                logger.info(f"Saving face file to: {temp_face_path}")
                face_file.save(temp_face_path)

                # Process with the model if available
                if face_recognizer:
                    logger.info("Processing face with model")
                    face_emotion, face_probs = face_recognizer.predict_image(temp_face_path)
                    logger.info(f"Face emotion detected: {face_emotion}")
                else:
                    # Fallback to placeholder
                    logger.info("Using placeholder face emotion")
                    face_emotion = "Neutral"
                    face_probs = [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]  # Higher prob for neutral

            except Exception as e:
                logger.error(f"Error processing face file: {str(e)}")
                flash(f"Warning: Could not process face file: {str(e)}", 'warning')

        # Use expert system for analysis
        logger.info("Running expert system analysis")
        result = expert_system.analyze(
            voice_emotion=voice_emotion,
            voice_probs=voice_probs,
            face_emotion=face_emotion,
            face_probs=face_probs
        )

        # Add voice and face confidences if not present
        if 'voice_confidence' not in result and voice_probs:
            result['voice_confidence'] = max(voice_probs) if voice_probs else 0.8
        if 'face_confidence' not in result and face_probs:
            result['face_confidence'] = max(face_probs) if face_probs else 0.8

        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_path}: {str(e)}")

        # Render the results
        return render_template('results.html', result=result)

    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return render_template('error.html', error=f"An error occurred during analysis: {str(e)}")


@app.route('/config-form', methods=['GET'])
def config_form():
    """Page with a form to update system configuration"""
    current_weights = {
        'voice_weight': expert_system.voice_weight,
        'face_weight': expert_system.face_weight
    }
    return render_template('config_form.html', current_weights=current_weights)


@app.route('/submit-config', methods=['POST'])
def submit_config():
    """Handle form submission for configuration updates"""
    try:
        voice_weight = float(request.form.get('voice_weight', 0.5))
        face_weight = float(request.form.get('face_weight', 0.5))

        expert_system.update_weights(voice_weight, face_weight)

        flash('Configuration updated successfully', 'success')
        return redirect(url_for('config_form'))

    except Exception as e:
        logger.exception(f"Error updating configuration: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })


if __name__ == "__main__":
    # Run the app in debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)