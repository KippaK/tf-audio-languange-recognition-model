import os

# Suppress TensorFlow unnecessary warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from colorama import init, Fore, Style

from config import LABELS, PREFIX_MAP

# Initialize colorama for Windows console coloring
init()

class LanguageDetector:
    """
    Audio file language detection using a pretrained neural network model.
    Handles audio preprocessing, mel-spectrogram conversion, and prediction.
    """

    def __init__(self, model_path="best_model_tf"):
        """Initializes the detector by loading the best model."""
        try:
            self.model = load_model(model_path)
            # Audio processing parameters (same as in training)
            self.sample_rate = 16000
            self.fixed_length = self.sample_rate * 2  # 2 seconds
            self.n_mels = 64  # Number of mel frequencies
            self.labels = LABELS
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_audio(self, audio):
        """
        Preprocesses audio for model input.
        - Converts stereo to mono
        - Normalizes amplitude
        - Standardizes length by padding/truncating
        """
        # Stereo -> mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Normalize volume
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio)) + 1e-9
        audio = audio / max_val

        # Pad or truncate to fixed length
        audio_len = len(audio)
        if audio_len > self.fixed_length:
            audio = audio[:self.fixed_length]
        else:
            padding = self.fixed_length - audio_len
            audio = np.pad(audio, (0, padding))

        return audio

    def audio_to_mel(self, audio):
        """
        Converts audio to mel-spectrogram.
        Uses the same process as model training.
        """
        # Convert to TensorFlow tensor
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

        # Calculate spectrogram (STFT)
        stft = tf.signal.stft(audio_tensor, frame_length=400, frame_step=160, fft_length=512)
        spectrogram = tf.abs(stft)

        # Create mel filter
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=257,
            sample_rate=self.sample_rate,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )

        # Convert to mel scale
        mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
        mel_spec = tf.math.log(mel_spec + 1e-6)  # Logarithmic compression

        # Normalize
        mean = tf.reduce_mean(mel_spec)
        std = tf.math.reduce_std(mel_spec)
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        # Add dimensions for model input: (batch, time, frequency, channel)
        mel_spec = tf.expand_dims(mel_spec, axis=0)   # Batch dimension
        mel_spec = tf.expand_dims(mel_spec, axis=-1)  # Channel dimension

        return mel_spec.numpy()

    def predict(self, audio_path):
        """
        Predicts the language of an audio file.

        Returns:
            tuple: (language, confidence) or (None, 0.0) on error
        """
        if self.model is None:
            return None, 0.0

        try:
            # Load audio file
            audio, sr = sf.read(audio_path)

            # Simple sample rate conversion
            if sr != self.sample_rate and sr > self.sample_rate:
                step = sr // self.sample_rate
                audio = audio[::step]

            # Preprocess audio
            audio = self.preprocess_audio(audio)

            # Convert to mel-spectrogram
            mel_spec = self.audio_to_mel(audio)

            # Make prediction
            predictions = self.model.predict(mel_spec, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])

            return self.labels[pred_idx], confidence

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, 0.0

def load_transcripts(transcript_file):
    """
    Loads transcripts from file.

    Returns:
        dict: Dictionary with filenames as keys and transcripts as values
    """
    transcripts = {}
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    filename, transcript = line.strip().split('\t', 1)
                    transcripts[filename] = transcript
        print(f"Loaded {len(transcripts)} transcripts")
    except Exception as e:
        print(f"Error loading transcripts: {e}")
    return transcripts

def main():
    """Tests the trained model with test files and displays results."""
    detector = LanguageDetector()
    if detector.model is None:
        return

    data_dir = "data/test"
    transcript_file = os.path.join(data_dir, "transcripts.txt")

    # Check that test directory exists
    if not os.path.exists(data_dir):
        print(f"Test directory not found: {data_dir}")
        return

    # Load transcripts
    transcripts = load_transcripts(transcript_file)

    # Find WAV files
    wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"No WAV files found in: {data_dir}")
        return

    print(f"Testing {len(wav_files)} files...")

    correct_predictions = 0

    # Process all test files
    for filename in sorted(wav_files):
        filepath = os.path.join(data_dir, filename)
        language, confidence = detector.predict(filepath)

        if language:
            # Determine true language based on filename
            true_language = None
            for prefix, lang in PREFIX_MAP.items():
                if filename.startswith(prefix):
                    true_language = lang
                    break

            # Get transcript for file
            transcript = transcripts.get(filename, "No transcript available")

            # Check if prediction is correct
            is_correct = (language == true_language) if true_language else None

            # Create result line with basic info
            result_line = f"{filename} | Prediction: {language} ({confidence*100:.1f}%)"

            if true_language:
                # Add color-coded result (green correct, red wrong)
                if is_correct:
                    status = f"{Fore.GREEN}CORRECT{Style.RESET_ALL}"
                    correct_predictions += 1
                else:
                    status = f"{Fore.RED}WRONG{Style.RESET_ALL}"

                result_line += f" | {status} (True: {true_language})"

            # Add transcript to result
            result_line += f" | Transcript: {transcript[:50]}{'...' if len(transcript) > 50 else ''}"
            print(result_line)

    # Display summary with colors
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)

    if wav_files:
        accuracy = (correct_predictions / len(wav_files)) * 100

        # Color-code overall accuracy
        if accuracy >= 80:
            accuracy_color = Fore.GREEN
        elif accuracy >= 60:
            accuracy_color = Fore.YELLOW
        else:
            accuracy_color = Fore.RED

        print(f"Accuracy: {accuracy_color}{correct_predictions}/{len(wav_files)} ({accuracy:.1f}%){Style.RESET_ALL}")

if __name__ == "__main__":
    main()
