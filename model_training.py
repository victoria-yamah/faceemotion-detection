"""
model_training.py
-----------------
This file handles:
1. Downloading or loading the FER2013 dataset
2. Converting CSV pixel data to images
3. Building and training a CNN model
4. Saving the trained model as face_emotionModel.h5
"""

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


IMAGE_DIR = "fer2013"  # Directory where images are stored
MODEL_PATH = "face_emotionModel.h5"


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    # data generators for train/test
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        validation_split=0.2  
    )

    train_gen = datagen.flow_from_directory(
        os.path.join(IMAGE_DIR, "train"),
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        os.path.join(IMAGE_DIR, "train"),
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        subset="validation"
    )

    model = build_model(input_shape=(48, 48, 1), num_classes=train_gen.num_classes)

    print("Starting model training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25
    )

    model.save(MODEL_PATH)
    print(f"Model training complete. Saved as {MODEL_PATH}")

    return history



if __name__ == "__main__":
    train_model()