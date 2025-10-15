import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape, num_classes):
    """
    Balanced CNN model - good performance with reasonable regularization.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),  # Moderate dropout
        
        # Second conv block  
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.35),  # Moderate dropout
        
        # Third conv block
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.45),  # Moderate dropout
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),   # Moderate dropout
        layers.Dense(num_classes, activation='softmax')
    ])

    # Balanced learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model