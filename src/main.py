import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from colorama import init, Fore, Style

# Alustetaan colorama Windows-konsolin väritystä varten
init()

class LanguageDetector:
    """
    Äänitiedostojen kielentunnistus esikoulutetulla neuroverkkomallilla.
    Hoitaa äänen esikäsittelyn, mel-spektrogrammin muunnoksen ja ennusteen.
    """
    
    def __init__(self, model_path="best_model.keras"):
        """Alustaa tunnistimen lataamalla koulutetun mallin."""
        try:
            self.model = load_model(model_path)
            # Äänenkäsittelyparametrit (samat kuin koulutuksessa)
            self.sample_rate = 16000
            self.fixed_length = self.sample_rate * 2  # 2 sekuntia
            self.n_mels = 64  # Mel-taajuuksien määrä
            self.labels = ["Finnish", "English"]  # Tunnistettavat kielet
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_audio(self, audio):
        """
        Esikäsittelee äänen mallin syötteeksi.
        - Muuntaa stereon monoksi
        - Normalisoi amplitudin
        - Tasaa pituuden täyttämällä/leikkaamalla
        """
        # Stereo -> mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalisoi äänenvoimakkuus
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio)) + 1e-9
        audio = audio / max_val
        
        # Täytä tai leikkaa vakiopituiseksi
        audio_len = len(audio)
        if audio_len > self.fixed_length:
            audio = audio[:self.fixed_length]
        else:
            padding = self.fixed_length - audio_len
            audio = np.pad(audio, (0, padding))
        
        return audio
    
    def audio_to_mel(self, audio):
        """
        Muuntaa äänen mel-spektrogrammiksi.
        Käyttää samaa prosessia kuin mallin koulutus.
        """
        # Muunna TensorFlow-tensoriksi
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Laske spektrogrammi (STFT)
        stft = tf.signal.stft(audio_tensor, frame_length=512, frame_step=256, fft_length=512)
        spectrogram = tf.abs(stft)
        
        # Luo mel-suodatin
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=257,
            sample_rate=self.sample_rate,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )
        
        # Muunna mel-asteikolle
        mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
        mel_spec = tf.math.log(mel_spec + 1e-6)  # Logaritminen kompressio
        
        # Normalisoi
        mean = tf.reduce_mean(mel_spec)
        std = tf.math.reduce_std(mel_spec)
        mel_spec = (mel_spec - mean) / (std + 1e-6)
        
        # Lisää ulottuvuudet mallin syötteelle: (batch, aika, taajuus, kanava)
        mel_spec = tf.expand_dims(mel_spec, axis=0)   # Batch-ulottuvuus
        mel_spec = tf.expand_dims(mel_spec, axis=-1)  # Kanava-ulottuvuus
        
        return mel_spec.numpy()
    
    def predict(self, audio_path):
        """
        Ennustaa äänitiedoston kielen.
        
        Palauttaa:
            tuple: (kieli, luottamus) tai (None, 0.0) virhetilanteessa
        """
        if self.model is None:
            return None, 0.0
            
        try:
            # Lataa äänitiedosto
            audio, sr = sf.read(audio_path)
            
            # Yksinkertainen näytteenottotaajuuden muutos
            if sr != self.sample_rate and sr > self.sample_rate:
                step = sr // self.sample_rate
                audio = audio[::step]
            
            # Esikäsittele ääni
            audio = self.preprocess_audio(audio)
            
            # Muunna mel-spektrogrammiksi
            mel_spec = self.audio_to_mel(audio)
            
            # Tee ennuste
            predictions = self.model.predict(mel_spec, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            
            return self.labels[pred_idx], confidence
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None, 0.0

def load_transcripts(transcript_file):
    """
    Lataa tekstitykset tiedostosta.
    
    Palauttaa:
        dict: Sanakirja, jossa tiedostonimet avaimina ja tekstitykset arvoina
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
    """Testaa koulutettua mallia testitiedostoilla ja näyttää tulokset."""
    detector = LanguageDetector()
    if detector.model is None:
        return
    
    data_dir = "data/test"
    transcript_file = os.path.join(data_dir, "transcripts.txt")
    
    # Tarkista että testihakemisto on olemassa
    if not os.path.exists(data_dir):
        print(f"Test directory not found: {data_dir}")
        return
    
    # Lataa tekstitykset
    transcripts = load_transcripts(transcript_file)
    
    # Etsi WAV-tiedostot
    wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in: {data_dir}")
        return
    
    print(f"Testing {len(wav_files)} files...")
    
    correct_predictions = 0
    
    # Käy läpi kaikki testitiedostot
    for filename in sorted(wav_files):
        filepath = os.path.join(data_dir, filename)
        language, confidence = detector.predict(filepath)
        
        if language:
            # Määritä todellinen kieli tiedostonimen perusteella
            true_language = None
            if filename.startswith('su_'):
                true_language = "Finnish"
            elif filename.startswith('en_'):
                true_language = "English"
            
            # Hae tekstitys tiedostolle
            transcript = transcripts.get(filename, "No transcript available")
            
            # Tarkista onko ennuste oikein
            is_correct = (language == true_language) if true_language else None
            
            # Luo tulosrivi perustiedoilla
            result_line = f"{filename} | Prediction: {language} ({confidence*100:.1f}%)"
            
            if true_language:
                # Lisää värikoodattu tulos (vihreä oikein, punainen väärin)
                if is_correct:
                    status = f"{Fore.GREEN}CORRECT{Style.RESET_ALL}"
                    correct_predictions += 1
                else:
                    status = f"{Fore.RED}WRONG{Style.RESET_ALL}"
                
                result_line += f" | {status} (True: {true_language})"
            
            # Lisää tekstitys tulokseen
            result_line += f" | Transcript: {transcript[:50]}{'...' if len(transcript) > 50 else ''}"
            print(result_line)
    
    # Näytä yhteenveto väreillä
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    if wav_files:
        accuracy = (correct_predictions / len(wav_files)) * 100
        
        # Värikoodaa kokonaistarkkuus
        if accuracy >= 80:
            accuracy_color = Fore.GREEN
        elif accuracy >= 60:
            accuracy_color = Fore.YELLOW
        else:
            accuracy_color = Fore.RED
            
        print(f"Accuracy: {accuracy_color}{correct_predictions}/{len(wav_files)} ({accuracy:.1f}%){Style.RESET_ALL}")

if __name__ == "__main__":
    main()
    