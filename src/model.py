import os

# Suppress TensorFlow oneDNN error messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers

def build_improved_model(input_shape, num_classes):
    """
    Refined convolutional neural network model for audio language detection.

    Improvements:
    - Optimized architecture
    - Attention mechanism
    - Better regularization
    """
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Second convolutional block
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Third convolutional block
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)

    # Fourth convolutional block
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Global Attention Pooling
    channel_axis = 3
    time_axis = 1

    # Channel attention
    channel_avg = layers.GlobalAveragePooling2D()(x)
    channel_max = layers.GlobalMaxPooling2D()(x)
    channel_attention = layers.Add()([layers.Dense(256)(channel_avg),
                                    layers.Dense(256)(channel_max)])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, 256))(channel_attention)
    x = layers.Multiply()([x, channel_attention])

    # Time-distributed features
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Optimized optimization
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_model(input_shape, num_classes):
    """For compatibility with old code"""
    return build_improved_model(input_shape, num_classes)