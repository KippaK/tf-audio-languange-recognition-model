import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
from prepare_training_data import create_mel_spectrogram

def test_model():
    """
    Test model performance on unseen test data.
    Provides comprehensive evaluation metrics.
    """
    print("Testing model performance...")
    
    try:
        # Load test data
        fi_test = tfds.load("xtreme_s/fleurs.fi_fi", split="test[:200]", as_supervised=True)
        en_test = tfds.load("xtreme_s/fleurs.en_us", split="test[:200]", as_supervised=True)
        
        # Process test data
        fi_test = fi_test.map(lambda x, y: (create_mel_spectrogram(x), 0))
        en_test = en_test.map(lambda x, y: (create_mel_spectrogram(x), 1))
        
        # Combine test datasets
        test_ds = tf.data.Dataset.sample_from_datasets([fi_test, en_test], [0.5, 0.5])
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Load model
        model = load_model("best_model.keras")
        
        # Evaluate
        print("Running evaluation...")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
        
        # Results
        print("\n" + "="*40)
        print("TEST RESULTS:")
        print("="*40)
        print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test loss: {test_loss:.4f}")
        
        # Performance assessment
        if test_accuracy >= 0.75:
            print("EXCELLENT generalization!")
        elif test_accuracy >= 0.65:
            print("GOOD generalization")
        elif test_accuracy >= 0.55:
            print("ACCEPTABLE generalization")
        else:
            print("Model needs improvement")
            
        return test_accuracy
        
    except Exception as e:
        print(f"Testing failed: {e}")
        return None

if __name__ == "__main__":
    test_model()