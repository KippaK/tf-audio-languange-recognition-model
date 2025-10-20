import os

# Estetään TensorFlowin turhat varoitukset
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from prepare_training_data import create_mel_spectrogram
import numpy as np
from colorama import init, Fore, Style

# Alustetaan colorama Windows-konsolin värejä varten
init()

def test_model():
    """
    Testaa koulutetun mallin suorituskykyä näkemättömällä testidatalla.
    """
    print("Testing model performance with test data...")
    
    try:
        # Ladataan testidata FLEURS-datasetistä
        fi_test = tfds.load("xtreme_s/fleurs.fi_fi", split="test", as_supervised=True)
        en_test = tfds.load("xtreme_s/fleurs.en_us", split="test", as_supervised=True)
        
        # Rajataan testidata kohtuulliseen kokoon
        fi_test = fi_test.take(200)
        en_test = en_test.take(200)
        
        print(f"Test samples - Finnish: {len(list(fi_test))}, English: {len(list(en_test))}")
        
        # Käsitellään testidata samalla tavalla kuin koulutusdata
        fi_test = fi_test.map(lambda x, y: (create_mel_spectrogram(x), 0))
        en_test = en_test.map(lambda x, y: (create_mel_spectrogram(x), 1))
        
        # Yhdistetään testidatasettit tasapainoisesti
        test_ds = tf.data.Dataset.sample_from_datasets([fi_test, en_test], [0.5, 0.5])
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Ladataan koulutettu malli levyltä
        model = load_model("best_model.keras")
        
        # Arvioidaan mallin suorituskyky testidatalla
        print("Running evaluation...")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        
        # Värikoodataan tarkkuus tulosten perusteella
        if test_accuracy >= 0.80:
            accuracy_color = Fore.GREEN
        elif test_accuracy >= 0.75:
            accuracy_color = Fore.YELLOW
        else:
            accuracy_color = Fore.RED
        
        # Tulostetaan testitulokset väreillä
        print(f"\n{'='*50}")
        print(f"TEST RESULTS:")
        print(f"{'='*50}")
        print(f"Test accuracy: {accuracy_color}{test_accuracy:.4f} ({test_accuracy*100:.2f}%){Style.RESET_ALL}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test samples total: {len(list(fi_test)) + len(list(en_test))}")
        
        # Arvioidaan mallin yleistävyys testitulosten perusteella
        print(f"\nGENERALIZATION ASSESSMENT:")
        if test_accuracy >= 0.80:
            print(f"{Fore.GREEN}EXCELLENT generalization{Style.RESET_ALL}")
        elif test_accuracy >= 0.75:
            print(f"{Fore.YELLOW}GOOD generalization{Style.RESET_ALL}")
        elif test_accuracy >= 0.70:
            print(f"{Fore.YELLOW}ACCEPTABLE generalization{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}POOR generalization{Style.RESET_ALL}")
            
        return test_accuracy
        
    except Exception as e:
        print(f"{Fore.RED}Testing failed: {e}{Style.RESET_ALL}")
        return None

if __name__ == "__main__":
    test_model()
    