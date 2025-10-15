import tensorflow as tf
import tensorflow_datasets as tfds
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# Configuration
SAMPLE_RATE = 16000
DURATION = 2
FIXED_LENGTH = SAMPLE_RATE * DURATION
BATCH_SIZE = 32
EPOCHS = 30
N_MELS = 64

def create_mel_spectrogram(audio):
    """Convert audio to mel spectrogram."""
    audio = tf.cast(audio, tf.float32)
    max_val = tf.reduce_max(tf.abs(audio)) + 1e-9
    audio = audio / max_val
    
    audio_len = tf.shape(audio)[0]
    audio = tf.cond(
        audio_len > FIXED_LENGTH,
        lambda: audio[:FIXED_LENGTH],
        lambda: tf.pad(audio, [[0, FIXED_LENGTH - audio_len]])
    )
    
    stft = tf.signal.stft(audio, frame_length=512, frame_step=256, fft_length=512)
    spectrogram = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=257,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    
    mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    
    mean = tf.reduce_mean(mel_spec)
    std = tf.math.reduce_std(mel_spec)
    mel_spec = (mel_spec - mean) / (std + 1e-6)
    
    mel_spec = tf.expand_dims(mel_spec, -1)
    
    return mel_spec

def load_data():
    """Load more data for better learning."""
    print("Loading data...")
    
    try:
        # Use more data for better generalization
        fi_train = tfds.load("xtreme_s/fleurs.fi_fi", split="train[:1500]", as_supervised=True)
        fi_val = tfds.load("xtreme_s/fleurs.fi_fi", split="train[1500:1800]", as_supervised=True)
        
        en_train = tfds.load("xtreme_s/fleurs.en_us", split="train[:1500]", as_supervised=True)
        en_val = tfds.load("xtreme_s/fleurs.en_us", split="train[1500:1800]", as_supervised=True)
        
        print("Data loaded successfully")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Process data
    fi_train = fi_train.map(lambda audio, text: (create_mel_spectrogram(audio), 0))
    fi_val = fi_val.map(lambda audio, text: (create_mel_spectrogram(audio), 0))
    
    en_train = en_train.map(lambda audio, text: (create_mel_spectrogram(audio), 1))
    en_val = en_val.map(lambda audio, text: (create_mel_spectrogram(audio), 1))
    
    # Combine datasets
    train_ds = tf.data.Dataset.sample_from_datasets([fi_train, en_train], [0.5, 0.5])
    val_ds = tf.data.Dataset.sample_from_datasets([fi_val, en_val], [0.5, 0.5])
    
    # Batch and optimize
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def main():
    print("AUDIO LANGUAGE DETECTION - BALANCED")
    print("=" * 40)
    
    train_ds, val_ds = load_data()
    
    if train_ds is None:
        print("Data loading failed")
        return
    
    for x, y in train_ds.take(1):
        input_shape = x.shape[1:]
        print(f"Input shape: {input_shape}")
        break
    
    print("Building model...")
    model = build_model(input_shape=input_shape, num_classes=2)
    
    # More patient callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save("trained_model.keras")
    print("Model saved: trained_model.keras")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\nTRAINING RESULTS:")
    print(f"Training accuracy: {final_train_acc:.4f}")
    print(f"Validation accuracy: {final_val_acc:.4f}")
    
    if final_val_acc > 0.75:
        print("EXCELLENT performance!")
    elif final_val_acc > 0.65:
        print("GOOD performance")
    elif final_val_acc > 0.55:
        print("ACCEPTABLE performance")
    else:
        print("Model needs improvement")

if __name__ == "__main__":
    main()