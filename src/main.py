import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class LanguageDetector:
    def __init__(self, model_path="best_model.keras"):
        """
        Language detector for audio files.
        Loads pre-trained model for predictions.
        """
        try:
            self.model = load_model(model_path)
            self.sample_rate = 16000
            self.fixed_length = self.sample_rate * 2  # 2 seconds
            self.n_mels = 64
            self.labels = ["Finnish", "English"]
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_audio(self, audio):
        """
        Preprocess audio for model input.
        Converts to mono and normalizes.
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio)) + 1e-9
        audio = audio / max_val
        
        # Pad or trim to fixed length
        if len(audio) > self.fixed_length:
            audio = audio[:self.fixed_length]
        else:
            padding = self.fixed_length - len(audio)
            audio = np.pad(audio, (0, padding))
        
        return audio
    
    def audio_to_mel(self, audio):
        """
        Convert audio to mel spectrogram.
        Uses same processing as training.
        """
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Create spectrogram
        stft = tf.signal.stft(audio_tensor, frame_length=512, frame_step=256, fft_length=512)
        spectrogram = tf.abs(stft)
        
        # Convert to mel scale
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=257,
            sample_rate=self.sample_rate,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )
        
        mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
        mel_spec = tf.math.log(mel_spec + 1e-6)
        
        # Normalize
        mean = tf.reduce_mean(mel_spec)
        std = tf.math.reduce_std(mel_spec)
        mel_spec = (mel_spec - mean) / (std + 1e-6)
        
        # Add dimensions for model input
        mel_spec = tf.expand_dims(mel_spec, axis=0)
        mel_spec = tf.expand_dims(mel_spec, axis=-1)
        
        return mel_spec.numpy()
    
    def predict(self, audio_path):
        """
        Predict language from audio file.
        Returns predicted language and confidence.
        """
        if self.model is None:
            return None, 0.0
            
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                if sr > self.sample_rate:
                    step = sr // self.sample_rate
                    audio = audio[::step]
            
            # Preprocess
            audio = self.preprocess_audio(audio)
            
            # Convert to mel spectrogram
            mel_spec = self.audio_to_mel(audio)
            
            # Predict
            predictions = self.model.predict(mel_spec, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            
            return self.labels[pred_idx], confidence
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, 0.0

def main():
    """
    Test the trained model on test files.
    Provides detailed performance analysis.
    """
    detector = LanguageDetector()
    if detector.model is None:
        return
    
    data_dir = "data/test"
    
    if not os.path.exists(data_dir):
        print(f"Test directory not found: {data_dir}")
        return
    
    # Find WAV files
    wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in: {data_dir}")
        return
    
    print(f"Testing {len(wav_files)} files...")
    
    results = []
    correct_predictions = 0
    
    for filename in sorted(wav_files):
        filepath = os.path.join(data_dir, filename)
        language, confidence = detector.predict(filepath)
        
        if language:
            # Determine true language from filename
            true_language = None
            if filename.startswith('su_'):
                true_language = "Finnish"
            elif filename.startswith('en_'):
                true_language = "English"
            
            # Check if prediction is correct
            is_correct = (language == true_language) if true_language else None
            
            # Create result line
            result_line = f"{filename} | Prediction: {language} ({confidence*100:.1f}%)"
            if true_language:
                status = "CORRECT" if is_correct else "WRONG"
                result_line += f" | {status} (True: {true_language})"
                
                if is_correct:
                    correct_predictions += 1
            
            print(result_line)
            results.append(result_line)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    if len(wav_files) > 0:
        accuracy = (correct_predictions / len(wav_files)) * 100
        print(f"Accuracy: {correct_predictions}/{len(wav_files)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    main()