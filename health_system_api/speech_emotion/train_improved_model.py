import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed to ensure reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# Define emotion labels
emotion_labels = {
    0: "female_angry", 1: "female_calm", 2: "female_fearful", 
    3: "female_happy", 4: "female_sad", 5: "male_angry", 
    6: "male_calm", 7: "male_fearful", 8: "male_happy", 9: "male_sad"
}

# Load features and labels
def load_data(data_path):
    """
    Load features and labels
    
    Parameters:
        data_path: data path
        
    Returns:
        X: features
        y: labels
    """
    X = np.load(os.path.join(data_path, "enhanced_features.npy"))
    y = np.load(os.path.join(data_path, "enhanced_labels.npy"))
    
    return X, y

# Prepare training data
def prepare_data(X, y, test_size=0.2, validation_size=0.2):
    """
    Prepare training, validation, and test data
    
    Parameters:
        X: features
        y: labels
        test_size: test set proportion
        validation_size: validation set proportion
        
    Returns:
        X_train, X_val, X_test: training, validation, and test features
        y_train, y_val, y_test: training, validation, and test labels
    """
    # First split out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split validation set from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=validation_size/(1-test_size), 
        random_state=42, 
        stratify=y_train_val
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=len(emotion_labels))
    y_val_cat = to_categorical(y_val, num_classes=len(emotion_labels))
    y_test_cat = to_categorical(y_test, num_classes=len(emotion_labels))
    
    print(f"Training set shape: {X_train.shape}, {y_train_cat.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val_cat.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test_cat.shape}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_test

# Build improved model
def build_improved_model(input_shape, num_classes):
    """
    Build improved model
    
    Parameters:
        input_shape: input shape
        num_classes: number of classes
        
    Returns:
        model: built model
    """
    model = Sequential()
    
    # Fully connected layers
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

# Train model
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, model_path=None):
    """
    Train model
    
    Parameters:
        model: model
        X_train: training features
        y_train: training labels
        X_val: validation features
        y_val: validation labels
        batch_size: batch size
        epochs: number of training epochs
        model_path: model save path
        
    Returns:
        history: training history
    """
    # Define callbacks
    callbacks = []
    
    # If model save path is provided, add ModelCheckpoint callback
    if model_path:
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Evaluate model
def evaluate_model(model, X_test, y_test_cat, y_test, class_names):
    """
    Evaluate model
    
    Parameters:
        model: model
        X_test: test features
        y_test_cat: test labels (one-hot encoded)
        y_test: test labels (original)
        class_names: class names
        
    Returns:
        accuracy: accuracy
    """
    # Evaluate model on test set
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test set accuracy: {accuracy*100:.2f}%")
    
    # Predict test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('/home/ubuntu/speech_emotion_analyzer/results/improved_confusion_matrix.png')
    plt.close()
    
    return accuracy

# Plot training history
def plot_history(history):
    """
    Plot training history
    
    Parameters:
        history: training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/speech_emotion_analyzer/results/improved_training_history.png')
    plt.close()

# Main function
def main():
    # Data path
    data_path = "/home/ubuntu/speech_emotion_analyzer/data/processed"
    
    # Model save path
    model_dir = "/home/ubuntu/speech_emotion_analyzer/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Results save path
    results_dir = "/home/ubuntu/speech_emotion_analyzer/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    X, y = load_data(data_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test_cat, y_test = prepare_data(X, y)
    
    # Get class names
    class_names = [emotion_labels[i] for i in range(len(emotion_labels))]
    
    # Build model
    input_shape = X_train.shape[1]
    num_classes = len(emotion_labels)
    model = build_improved_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Train model
    model_path = os.path.join(model_dir, "improved_emotion_model.h5")
    history = train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, model_path=model_path)
    
    # Plot training history
    plot_history(history)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test_cat, y_test, class_names)
    
    # Save final model
    model.save(os.path.join(model_dir, "improved_emotion_model_final.h5"))
    
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(os.path.join(model_dir, "improved_model.json"), "w") as json_file:
        json_file.write(model_json)
    
    # Save accuracy
    with open(os.path.join(results_dir, "improved_accuracy.txt"), "w") as f:
        f.write(f"Test set accuracy: {accuracy*100:.2f}%")
    
    print(f"Model training and evaluation complete, final accuracy: {accuracy*100:.2f}%")
    print(f"Model and results saved to {model_dir} and {results_dir}")

if __name__ == "__main__":
    main()