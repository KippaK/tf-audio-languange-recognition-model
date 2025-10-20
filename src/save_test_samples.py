import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import soundfile as sf
import numpy as np

def save_test_samples(num_samples=10, output_dir="data/test"):
    """
    Tallentaa testinäytteitä tiedostoihin mallin testaamista varten.
    
    Luo tasapainoisen testijoukon suomen ja englannin kielisistä ääninäytteistä
    käyttäen FLEURS-datasetin OIKEAA testidataa (test-split). Nämä tiedostot 
    voidaan myöhemmin käyttää manuaaliseen testaamiseen erillisen testausskriptin kautta.
    """
    # Luodaan kohdehakemisto jos sitä ei ole
    os.makedirs(output_dir, exist_ok=True)
    
    # Määritellään ladattavat kielidatasettit - käytetään TEST-dataa
    datasets = {
        "finnish": ("xtreme_s/fleurs.fi_fi", "test"),  # Suomenkielinen testidata
        "english": ("xtreme_s/fleurs.en_us", "test")   # Englanninkielinen testidata
    }
    
    # Luodaan tekstitystiedosto, joka sisältää tiedostojen nimet ja puheen sisällön
    transcript_file = os.path.join(output_dir, "transcripts.txt")
    
    print(f"Saving {num_samples} test samples per language...")
    
    # Avataan tekstitystiedosto kirjoitusta varten
    with open(transcript_file, "w", encoding="utf-8") as f:
        # Käsitellään kumpikin kieli vuorollaan
        for lang, (dataset_name, split) in datasets.items():
            print(f"Loading {lang} test data...")
            
            try:
                # Ladataan pieni osa datasetin TEST-osuudesta
                ds = tfds.load(dataset_name, split=f"{split}[:{num_samples}]", as_supervised=True)
                
                # Käsitellään jokainen ääninäyte datasetissä
                for i, (audio, text) in enumerate(ds):
                    # Muunnetaan TensorFlow-tensor numpy-taulukoksi
                    audio_array = audio.numpy()
                    
                    # Muunnetaan stereoääni monoksi tarvittaessa
                    # Lasketaan kanavien keskiarvo jos ääni on stereo
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Normalisoidaan äänen amplitudi välille [-1, 1]
                    audio_array = audio_array.astype(np.float32)
                    max_val = np.max(np.abs(audio_array)) + 1e-9  # Vältetään nollalla jakaminen
                    audio_array = audio_array / max_val
                    
                    # Luodaan standardoitu tiedostonimi
                    # su_001.wav, su_002.wav, ... suomenkielisille
                    # en_001.wav, en_002.wav, ... englanninkielisille
                    prefix = "su" if lang == "finnish" else "en"
                    filename = f"{prefix}_{i+1:03d}.wav"  # 3-numeroinen indeksi etunollilla
                    filepath = os.path.join(output_dir, filename)
                    
                    # Tallennetaan äänitiedosto WAV-muodossa
                    # Käytetään 16kHz näytteenottotaajuutta ja 32-bit float formaattia
                    sf.write(filepath, audio_array, 16000, subtype='FLOAT')
                    
                    # Tallennetaan tekstitys tiedostoon
                    # Dekoodataan bytes -> utf-8 merkkijono
                    transcript = text.numpy().decode('utf-8')
                    f.write(f"{filename}\t{transcript}\n")
                    
                    print(f"Saved: {filename}")
                    
            except Exception as e:
                # Käsitellään mahdolliset virheet datan latauksessa
                print(f"Error loading {lang} test data: {e}")
    
    # Tulostetaan yhteenveto tallennetusta datasta
    print(f"All {num_samples * len(datasets)} test samples saved to: {output_dir}")
    print(f"Transcripts saved to: {transcript_file}")

if __name__ == "__main__":
    # Suoritetaan testinäytteiden tallennus oletusarvoilla
    # 10 näytettä per kieli, tallennushakemistona "data/test"
    save_test_samples(num_samples=10, output_dir="data/test")
    