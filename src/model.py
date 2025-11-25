import os

# Suppress TensorFlow oneDNN error messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers

def build_improved_model(input_shape, num_classes):
    """
    Simplified convolutional neural network for audio language detection.

    Simplified architecture:
    - Removed 4th conv block (256 filters was overkill)
    - Smaller dense layers
    - Less complexity = better generalization with limited data
    """
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.15)(x)  # Reverted to first attempt

    # Second convolutional block
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)  # Reverted to first attempt

    # Third convolutional block
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)  # Reverted to first attempt

    # REMOVED 4th conv block - was too complex for this task

    # Global Average Pooling - simpler than attention mechanism
    x = layers.GlobalAveragePooling2D()(x)

    # Simplified dense layers - fewer parameters
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Reduced from 512
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)  # Reverted to first attempt

    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Reduced from 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Reverted to first attemptlayer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Standard optimizer - no label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_model(input_shape, num_classes):
    """For compatibility with old code"""
    return build_improved_model(input_shape, num_classes)