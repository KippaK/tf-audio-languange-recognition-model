import os

# Suppress TensorFlow unnecessary warnings
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
from config import LANGUAGES

# Initialize colorama for Windows console colors
init()

def test_model():
    """
    Tests the trained model's performance with unseen test data.
    """
    print("Testing model performance with test data...")

    try:
        test_datasets = []
        label_map = {}

        # Load and process test data for each language
        for i, lang in enumerate(LANGUAGES):
            dataset_name = lang["dataset"]
            label = i
            ds = tfds.load(dataset_name, split="test", as_supervised=True)

            # Limit test data to a reasonable size
            take_count = 200  # you can modify this as needed
            ds = ds.take(take_count)

            # Add label and convert to mel-spectrogram
            ds = ds.map(lambda x, y: (create_mel_spectrogram(x), label))

            test_datasets.append(ds)
            label_map[i] = lang["name"]

        # Combine test datasets in a balanced way
        test_ds = tf.data.Dataset.sample_from_datasets(test_datasets)
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        # Load the trained model from disk
        model = load_model("best_model_tf")

        # Evaluate model performance on test data
        print("Running evaluation...")
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

        # --- Calculate confusion matrix ---
        print("\nCalculating confusion matrix...")

        # 1. Get true labels
        y_true = []
        for _, y in test_ds:
            y_true.extend(y.numpy())
        y_true = np.array(y_true)

        # 2. Model predictions
        y_pred_prob = model.predict(test_ds)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # 3. Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        print("\nConfusion matrix:")
        print(cm)

        # 4. (Optional) Print for labeled table
        print("\nLanguages (label index -> name):")
        for idx, name in label_map.items():
            print(f"{idx}: {name}")

        # 5. (Optional) Draw heatmap if desired
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                        xticklabels=[label_map[i] for i in range(len(label_map))],
                        yticklabels=[label_map[i] for i in range(len(label_map))])
            plt.xlabel("Ennustettu")
            plt.ylabel("Todellinen")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()
        except Exception:
            print("Could not draw heatmap (matplotlib/seaborn missing)")


        # Color-code accuracy based on results
        if test_accuracy >= 0.80:
            accuracy_color = Fore.GREEN
        elif test_accuracy >= 0.75:
            accuracy_color = Fore.YELLOW
        else:
            accuracy_color = Fore.RED

        # Print test results with colors
        total_samples = sum([len(list(ds)) for ds in test_datasets])
        print(f"\n{'='*50}")
        print(f"TEST RESULTS:")
        print(f"{'='*50}")
        print(f"Test accuracy: {accuracy_color}{test_accuracy:.4f} ({test_accuracy*100:.2f}%){Style.RESET_ALL}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test samples total: {total_samples}")

        # Assess model generalization
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
