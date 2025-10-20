import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import soundfile as sf
import numpy as np

def save_test_samples(num_samples=10, output_dir="data/test"):
    """
    Save test samples to files for model testing.
    Creates balanced dataset of Finnish and English audio.
    """
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        "finnish": ("xtreme_s/fleurs.fi_fi", "train"),
        "english": ("xtreme_s/fleurs.en_us", "train")
    }
    
    transcript_file = os.path.join(output_dir, "transcripts.txt")
    
    print(f"Saving {num_samples} test samples per language...")
    
    with open(transcript_file, "w", encoding="utf-8") as f:
        for lang, (dataset_name, split) in datasets.items():
            print(f"Loading {lang} data...")
            
            try:
                # Load small subset
                ds = tfds.load(dataset_name, split=f"{split}[:{num_samples}]", as_supervised=True)
                
                for i, (audio, text) in enumerate(ds):
                    # Convert audio
                    audio_array = audio.numpy()
                    
                    # Convert to mono if needed
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Normalize
                    audio_array = audio_array.astype(np.float32)
                    max_val = np.max(np.abs(audio_array)) + 1e-9
                    audio_array = audio_array / max_val
                    
                    # Create filename
                    prefix = "su" if lang == "finnish" else "en"
                    filename = f"{prefix}_{i+1:03d}.wav"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save as WAV
                    sf.write(filepath, audio_array, 16000, subtype='FLOAT')
                    
                    # Save transcript
                    transcript = text.numpy().decode('utf-8')
                    f.write(f"{filename}\t{transcript}\n")
                    
                    print(f"Saved: {filename}")
                    
            except Exception as e:
                print(f"Error loading {lang}: {e}")
    
    print(f"All {num_samples * len(datasets)} test samples saved to: {output_dir}")
    print(f"Transcripts saved to: {transcript_file}")

if __name__ == "__main__":
    save_test_samples(num_samples=10, output_dir="data/test")