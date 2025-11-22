"Author: Zhaozhe Zhang"


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data increase 
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory('fer2013/', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory('fer2013/', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical', subset='validation')

# CNN model node set
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 类情绪
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=25, validation_data=val_data)
model.save('face_emotion_model.h5')
