import os

# Suppress TensorFlow unnecessary warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import soundfile as sf
import numpy as np

from config import LANGUAGES

def save_test_samples(num_samples=10, output_dir="data/test"):
    """
    Saves test samples to files for model testing.

    This uses the LANGUAGES array from config.py, so new languages
    can be added without modifying this script. Each language
    definition should have:
        - prefix: e.g. \"su_\", \"en_\"
        - name: language name
        - dataset: TFDS dataset name
        - split: split to use (e.g. \"test\")
    """

    # Create target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create transcript file
    transcript_file = os.path.join(output_dir, "transcripts.txt")

    print(f"Saving {num_samples} test audio files per language...")

    with open(transcript_file, "w", encoding="utf-8") as f:

        # Process languages from config
        for lang in LANGUAGES:
            name = lang["name"]
            prefix = lang["prefix"]
            dataset_name = lang["dataset"]
            split = "test"

            print(f"Loading {name} test data ({dataset_name})...")

            try:
                # Load a small portion of test data
                ds = tfds.load(dataset_name, split=f"{split}[:{num_samples}]", as_supervised=True)

                for i, (audio, text) in enumerate(ds):
                    audio_array = audio.numpy()

                    # Stereo -> mono
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)

                    # Normalize
                    audio_array = audio_array.astype(np.float32)
                    max_val = np.max(np.abs(audio_array)) + 1e-9
                    audio_array = audio_array / max_val

                    # Create filename e.g. su_001.wav
                    filename = f"{prefix}{i+1:03d}.wav"
                    filepath = os.path.join(output_dir, filename)

                    # Save wav
                    sf.write(filepath, audio_array, 16000, subtype="FLOAT")

                    # Transcript
                    transcript = text.numpy().decode("utf-8")
                    f.write(f"{filename}\t{transcript}\n")

                    print(f"Saved: {filename}")

            except Exception as e:
                print(f"Error loading {name} test data: {e}")

    print(f"\nSave complete: {num_samples * len(LANGUAGES)} files saved to directory: {output_dir}")
    print(f"Transcripts saved: {transcript_file}")

if __name__ == "__main__":
    # Execute test sample saving with default values
    # 10 samples per language, save directory "data/test"
    save_test_samples(num_samples=10, output_dir="data/test")
