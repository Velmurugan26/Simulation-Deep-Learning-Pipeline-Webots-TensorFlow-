# model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape=(64,64,3), num_classes=10):
    """
    Create a Convolutional Neural Network for object classification.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of object classes

    Returns:
        model (tf.keras.Model): Compiled CNN model
    """
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        # Convolutional Layer 3
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
