import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import numpy as np
import shutil
import time

from config import LANGUAGES, LANGUAGE_WEIGHTS

SAMPLE_RATE = 16000
DURATION = 2
FIXED_LENGTH = SAMPLE_RATE * DURATION
BATCH_SIZE = 48
EPOCHS = 50
N_MELS = 64

def simple_augment_audio(audio):
    """
    Kevyt augmentaatio: gain + satunnainen kohina.
    """
    if tf.random.uniform([]) > 0.5:
        gain = tf.random.uniform([], 0.9, 1.1)
        audio = audio * gain

    if tf.random.uniform([]) > 0.8:
        noise_std = tf.random.uniform([], 0.001, 0.005)
        noise = tf.random.normal(tf.shape(audio), stddev=noise_std)
        audio = audio + noise

    return audio

def create_mel_spectrogram(audio, augment=False):
    """
    Muuntaa 1D-audion mel-spektrogrammiksi, palauttaa shape (time, mel, 1).
    """
    audio = tf.cast(audio, tf.float32)
    max_val = tf.reduce_max(tf.abs(audio)) + 1e-9
    audio = audio / max_val

    audio_len = tf.shape(audio)[0]
    audio = tf.cond(
        audio_len > FIXED_LENGTH,
        lambda: audio[:FIXED_LENGTH],
        lambda: tf.pad(audio, [[0, tf.maximum(FIXED_LENGTH - audio_len, 0)]])
    )

    if augment:
        audio = simple_augment_audio(audio)

    stft = tf.signal.stft(audio, frame_length=400, frame_step=160, fft_length=512)
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
    """
    Lataa datan konfiguraation LANGUAGES-listasta.
    Varmistaa, ettei pyydetä suurempaa slicea kuin datasetissä on.
    Palauttaa train_ds ja val_ds (tf.data.Dataset).
    """
    print("Loading data...")

    train_datasets = []
    val_datasets = []

    try:
        for label_index, lang in enumerate(LANGUAGES):
            name = lang["name"]
            dataset = lang["dataset"]

            builder = tfds.builder(dataset)
            builder.download_and_prepare()  # varmistaa että info on saatavilla
            info = builder.info

            train_available = info.splits['train'].num_examples
            val_available = info.splits['validation'].num_examples

            train_take = min(train_available, lang.get('train_split', train_available))
            val_take = min(val_available, lang.get('val_split', val_available))

            train_ds = tfds.load(dataset, split=f"train[:{train_take}]", as_supervised=True)
            val_ds = tfds.load(dataset, split=f"validation[:{val_take}]", as_supervised=True)

            print(f"Loaded {name}: {train_take} train, {val_take} val (available train {train_available}, val {val_available})")

            train_ds = train_ds.map(
                lambda audio, text, li=label_index: (create_mel_spectrogram(audio, augment=True), li),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_ds = val_ds.map(
                lambda audio, text, li=label_index: (create_mel_spectrogram(audio, augment=False), li),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    if len(train_datasets) == 0:
        print("No training datasets loaded.")
        return None, None

    num_langs = len(train_datasets)
    weights = [1.0 / num_langs] * num_langs

    train_ds = tf.data.Dataset.sample_from_datasets(train_datasets, weights=weights)
    val_ds = tf.data.Dataset.sample_from_datasets(val_datasets, weights=weights)

    train_ds = (train_ds
                .cache()
                .shuffle(1500, reshuffle_each_iteration=True)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .cache()
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds

class SmartTrainingCallback(Callback):
    """
    Callback joka kerää epoch-aikoja ja näyttää arvioidun jäljellä olevan ajan.
    """
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        self.total_epochs = self.params.get('epochs', EPOCHS)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

        avg_epoch_time = float(np.mean(self.epoch_times))
        remaining = avg_epoch_time * (self.total_epochs - (epoch + 1))

        rem_h = int(remaining // 3600)
        rem_m = int((remaining % 3600) // 60)

        print(f"\nEpoch {epoch+1}/{self.total_epochs}")
        print(f"   Time this epoch: {epoch_time:.1f}s")
        print(f"   Avg epoch time: {avg_epoch_time:.1f}s")
        print(f"   Estimated remaining: {rem_h}h {rem_m}min")

def main():
    for model_file in ['best_model_tf', 'trained_tf', 'best_model.keras', 'trained_model.keras']:
        if os.path.exists(model_file):
            try:
                if os.path.isdir(model_file):
                    shutil.rmtree(model_file)
                else:
                    os.remove(model_file)
                print(f"Removed previous {model_file}")
            except Exception:
                pass

    print("AUDIO LANGUAGE DETECTION")
    print("=" * 60)
    print(f"Training {len(LANGUAGES)} languages: {[lang['name'] for lang in LANGUAGES]}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Class weights: {LANGUAGE_WEIGHTS}")

    train_ds, val_ds = load_data()
    if train_ds is None:
        print("Data loading failed")
        return

    for x, y in train_ds.take(1):
        input_shape = x.shape[1:]
        print(f"Input shape: {input_shape}")
        break

    print("Building model...")
    model = build_model(input_shape=input_shape, num_classes=len(LANGUAGES))

    smart_cb = SmartTrainingCallback()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5, 
            restore_best_weights=True,
            verbose=1,
            min_delta=0.002
        ),
        ModelCheckpoint(
            'best_model_tf',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_format='tf'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=6,
            min_lr=1e-5,
            verbose=1,
            cooldown=2
        ),
        smart_cb
    ]

    print("Starting training...")

    if tf.config.list_physical_devices('GPU'):
        print("GPU detected - using hardware acceleration")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=LANGUAGE_WEIGHTS,
        verbose=1
    )

    model.save("trained_tf", save_format='tf')
    print("Model saved: trained_tf")

    total_time = time.time() - getattr(smart_cb, 'start_time', time.time())

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    train_acc = history.history.get('accuracy', [0])[-1] if 'accuracy' in history.history else 0.0

    print("\n" + "=" * 60)
    print("TRAINING RESULTS:")
    print("=" * 60)
    print(f"Final training accuracy: {train_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {total_time/60:.1f}min")

    if best_val_acc > 0.75:
        print("EXCELLENT performance! (>75%)")
    elif best_val_acc > 0.65:
        print("VERY GOOD performance (65-75%)")
    elif best_val_acc > 0.55:
        print("GOOD performance (55-65%)")
    elif best_val_acc > 0.45:
        print("ACCEPTABLE performance (45-55%)")
    else:
        print("Needs improvement (<45%)")

if __name__ == "__main__":
    main()
