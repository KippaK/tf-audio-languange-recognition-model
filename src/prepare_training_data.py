import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import shutil

# Konfiguraatioparametrit
SAMPLE_RATE = 16000
DURATION = 2
FIXED_LENGTH = SAMPLE_RATE * DURATION
BATCH_SIZE = 32
EPOCHS = 50
N_MELS = 64

def augment_audio(audio):
    """
    Tehostettu data-augmentaatio äänelle.
    """
    original_audio = audio
    
    # 1. Kohina
    if tf.random.uniform([]) > 0.6:
        noise_std = tf.random.uniform([], 0.001, 0.02)
        noise = tf.random.normal(tf.shape(audio), stddev=noise_std)
        audio = audio + noise
    
    # 2. Vahvistuksen muutos
    if tf.random.uniform([]) > 0.6:
        gain = tf.random.uniform([], 0.6, 1.4)
        audio = audio * gain
    
    # 3. Frekvenssifiltteri simulointi
    if tf.random.uniform([]) > 0.7:
        # Yksinkertainen high-pass/low-pass efekti
        filter_type = tf.random.uniform([], 0, 1)
        if filter_type > 0.5:
            # High-pass efekti (vähennetään matalia taajuuksia)
            audio = audio * tf.random.uniform([], 1.0, 1.3)
        else:
            # Low-pass efekti (vähennetään korkeita taajuuksia)
            audio = audio * tf.random.uniform([], 0.7, 1.0)
    
    # 4. Aikavenytys/nopeutus käyttäen FFT:ta
    if tf.random.uniform([]) > 0.8:
        rate = tf.random.uniform([], 0.8, 1.2)
        current_length = tf.shape(audio)[0]
        new_length = tf.cast(tf.cast(current_length, tf.float32) * rate, tf.int32)
        
        # Resample FFT:llä
        audio_fft = tf.signal.fft(tf.cast(audio, tf.complex64))
        
        if rate > 1.0:  # Nopeutus
            # Leikkaa korkeimmat taajuudet
            keep_freqs = tf.cast(tf.cast(tf.shape(audio_fft)[0], tf.float32) / rate, tf.int32)
            audio_fft = audio_fft[:keep_freqs]
            # Täytä nollilla
            padding = tf.zeros([new_length - keep_freqs], dtype=tf.complex64)
            audio_fft = tf.concat([audio_fft, padding], axis=0)
        else:  # Hidastus
            # Täytä nollilla keskitaajuuksien väliin
            orig_freqs = tf.shape(audio_fft)[0]
            new_freqs = tf.cast(tf.cast(orig_freqs, tf.float32) * rate, tf.int32)
            keep_freqs = new_freqs // 2
            
            # Pidä alku- ja lopputaajuudet
            first_half = audio_fft[:keep_freqs]
            second_half = audio_fft[orig_freqs - keep_freqs:]
            zeros = tf.zeros([new_freqs - 2 * keep_freqs], dtype=tf.complex64)
            audio_fft = tf.concat([first_half, zeros, second_half], axis=0)
        
        audio = tf.math.real(tf.signal.ifft(audio_fft))
        audio = tf.reshape(audio, [new_length])
    
    # 5. Varmista että ääni pysyy vakiopituisena
    audio_len = tf.shape(audio)[0]
    audio = tf.cond(
        audio_len > FIXED_LENGTH,
        lambda: audio[:FIXED_LENGTH],
        lambda: tf.pad(audio, [[0, FIXED_LENGTH - audio_len]])
    )
    
    # 6. Rajaa äänenvoimakkuus
    audio = tf.clip_by_value(audio, -1.0, 1.0)
    
    return audio
def create_mel_spectrogram(audio, augment=False):
    """
    Parannettu mel-spektrogrammin luonti data-augmentaatiolla.
    """
    # Muunna float32:ksi ja normalisoi amplitudi
    audio = tf.cast(audio, tf.float32)
    max_val = tf.reduce_max(tf.abs(audio)) + 1e-9
    audio = audio / max_val
    
    # Täytä tai leikkaa ääni vakiopituiseksi
    audio_len = tf.shape(audio)[0]
    audio = tf.cond(
        audio_len > FIXED_LENGTH,
        lambda: audio[:FIXED_LENGTH],
        lambda: tf.pad(audio, [[0, FIXED_LENGTH - audio_len]])
    )
    
    # Data-augmentaatio vain koulutusdatalle
    if augment:
        audio = augment_audio(audio)
    
    # Laske lyhytaikainen Fourier-muunnos
    stft = tf.signal.stft(audio, frame_length=512, frame_step=256, fft_length=512)
    spectrogram = tf.abs(stft)
    
    # Luo mel-suodatinmatriisi
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=257,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    
    # Muunna mel-asteikolle
    mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
    mel_spec = tf.math.log(mel_spec + 1e-6)
    
    # Normalisoi spektrogrammi
    mean = tf.reduce_mean(mel_spec)
    std = tf.math.reduce_std(mel_spec)
    mel_spec = (mel_spec - mean) / (std + 1e-6)
    
    # Lisää kanavaulottuvuus
    mel_spec = tf.expand_dims(mel_spec, -1)
    
    return mel_spec

def load_data():
    """
    Parannettu datan lataus paremmalla tasapainolla.
    """
    print("Loading data...")
    
    try:
        # Käytetään vähemmän dataa mutta paremmalla tasapainolla
        fi_train = tfds.load("xtreme_s/fleurs.fi_fi", split="train[:1200]", as_supervised=True)
        fi_val = tfds.load("xtreme_s/fleurs.fi_fi", split="validation[:300]", as_supervised=True)
        
        en_train = tfds.load("xtreme_s/fleurs.en_us", split="train[:1200]", as_supervised=True)
        en_val = tfds.load("xtreme_s/fleurs.en_us", split="validation[:300]", as_supervised=True)
        
        print("Data loaded successfully")
        print(f"Finnish - Train: {len(list(fi_train))}, Val: {len(list(fi_val))}")
        print(f"English - Train: {len(list(en_train))}, Val: {len(list(en_val))}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Koulutusdataan augmentaatio, validointiin ei
    fi_train = fi_train.map(lambda audio, text: (create_mel_spectrogram(audio, augment=True), 0))
    fi_val = fi_val.map(lambda audio, text: (create_mel_spectrogram(audio, augment=False), 0))
    
    en_train = en_train.map(lambda audio, text: (create_mel_spectrogram(audio, augment=True), 1))
    en_val = en_val.map(lambda audio, text: (create_mel_spectrogram(audio, augment=False), 1))
    
    # Yhdistä datasettit TÄYSIN tasapainoisesti
    train_ds = tf.data.Dataset.sample_from_datasets([fi_train, en_train], weights=[0.5, 0.5])
    val_ds = tf.data.Dataset.sample_from_datasets([fi_val, en_val], weights=[0.5, 0.5])
    
    # Optimoi dataputki
    train_ds = train_ds.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def main():
    """Päivitetty pääfunktio class weighteilla."""
    
    # Poistaa edelliset mallit
    for model_file in ['best_model.keras', 'trained_model.keras']:
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"Removed previous {model_file}")

    print("AUDIO LANGUAGE DETECTION")
    print("=" * 50)
    
    # Lataa ja valmistele data
    train_ds, val_ds = load_data()
    
    if train_ds is None:
        print("Data loading failed")
        return
    
    # Hae mallin syöteformaatti
    for x, y in train_ds.take(1):
        input_shape = x.shape[1:]
        print(f"Input shape: {input_shape}")
        break
    
    # Rakenna parannettu malli
    print("Building model...")
    model = build_model(input_shape=input_shape, num_classes=2)
    
    # Class weights tasapainottamaan biasia
    # Koska suomi (0) menee useammin väärin, annetaan sille enemmän painoa
    class_weight = {0: 1.2, 1: 0.8}  # Suomi saa 20% enemmän painoa
    
    # Parannetut callbackit
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1)
    ]
    
    # Kouluta malli class weighteilla
    print("Starting training with class weights...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,  # Lisätty class weight
        verbose=1
    )
    
    # Tallenna lopullinen malli
    model.save("trained_model.keras")
    print("Model saved: trained_model.keras")
    
    # Varmista että paras malli on olemassa
    if not os.path.exists('best_model.keras'):
        print("Creating best_model.keras from trained model...")
        shutil.copy('trained_model.keras', 'best_model.keras')
        print("Best model created: best_model.keras")
    else:
        print("Best model already exists: best_model.keras")
    
    # Analysoi tulokset
    best_val_acc = max(history.history['val_accuracy'])
    train_acc = history.history['accuracy'][-1]
    
    print("\nTRAINING RESULTS:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Suorituskyvyn arviointi
    if best_val_acc > 0.80:
        print("EXCELLENT performance!")
    elif best_val_acc > 0.75:
        print("VERY GOOD performance")
    elif best_val_acc > 0.70:
        print("GOOD performance")
    elif best_val_acc > 0.65:
        print("ACCEPTABLE performance")
    else:
        print("Model needs improvement")

if __name__ == "__main__":
    main()
    