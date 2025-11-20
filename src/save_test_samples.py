import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import soundfile as sf
import numpy as np

from config import LANGUAGES

def save_test_samples(num_samples=10, output_dir="data/test"):
    """
    Tallentaa testinäytteitä tiedostoihin mallin testaamista varten.

    Tämä käyttää config.py-tiedoston LANGUAGES-taulukkoa, joten uusia kieliä
    voidaan lisätä muokkaamatta tätä skriptiä. Jokaisessa kielen 
    määrittelyssä tulee olla:
        - prefix: esim. \"su_\", \"en_\"
        - name: kielen nimi
        - dataset: TFDS-datasetin nimi
        - split: käytettävä split (esim. \"test\")
    """

    # Luodaan kohdehakemisto jos sitä ei ole
    os.makedirs(output_dir, exist_ok=True)

    # Luodaan tekstitystiedosto
    transcript_file = os.path.join(output_dir, "transcripts.txt")

    print(f"Tallennetaan {num_samples} testiääntä per kieli...")

    with open(transcript_file, "w", encoding="utf-8") as f:

        # Käydään configin kielet läpi
        for lang in LANGUAGES:
            name = lang["name"]
            prefix = lang["prefix"]
            dataset_name = lang["dataset"]
            split = "test"

            print(f"Ladataan {name} testidataa ({dataset_name})...")

            try:
                # Ladataan pieni osa testidatasta
                ds = tfds.load(dataset_name, split=f"{split}[:{num_samples}]", as_supervised=True)

                for i, (audio, text) in enumerate(ds):
                    audio_array = audio.numpy()

                    # Stereo -> mono
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)

                    # Normalisoidaan
                    audio_array = audio_array.astype(np.float32)
                    max_val = np.max(np.abs(audio_array)) + 1e-9
                    audio_array = audio_array / max_val

                    # Luodaan tiedostonimi esim. su_001.wav
                    filename = f"{prefix}{i+1:03d}.wav"
                    filepath = os.path.join(output_dir, filename)

                    # Tallennetaan wav
                    sf.write(filepath, audio_array, 16000, subtype="FLOAT")

                    # Tekstitys
                    transcript = text.numpy().decode("utf-8")
                    f.write(f"{filename}\t{transcript}\n")

                    print(f"Saved: {filename}")

            except Exception as e:
                print(f"Virhe ladattaessa {name} testidataa: {e}")

    print(f"\nTallennus valmis: {num_samples * len(LANGUAGES)} tiedostoa tallennettu hakemistoon: {output_dir}")
    print(f"Tekstitykset tallennettu: {transcript_file}")

if __name__ == "__main__":
    # Suoritetaan testinäytteiden tallennus oletusarvoilla
    # 10 näytettä per kieli, tallennushakemistona "data/test"
    save_test_samples(num_samples=10, output_dir="data/test")
    