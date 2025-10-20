# Spoken Language Identification
* [Objective](#objective)
* [Available models and languages](#available-models-and-languages)
* [Environment Setup](#environment-setup)
* [Gantt chart for project planning](#gantt-chart-for-project-planning)
* [File Structure](#file-structure)
* [How It Works](#how-it-works)
* [File Details](#file-details)
* [Usage Instructions](#usage-instructions)
* [Expected Results](#expected-results)

## Objective 
Spoken Language Identification (LID) is defined as detecting language from an audio clip by an unknown speaker, regardless of gender, manner of speaking, and distinct age speaker. Deep learning system for classifying audio as Finnish or English using CNN and mel spectrograms.

## Available models and languages
**FLEURS dataset** downloads can be found here: [Downloads](https://www.tensorflow.org/datasets/catalog/xtreme_s)

## Environment Setup
The models are implemented in PyTorch.
To use all of the functionality of the library, you should have:</br>
pytorch==2.8.0</br>
numpy==2.4.0</br>
tensorflow-datasets==4.9.9</br>
soundfile==0.13.1</br>

## Gantt chart for project planning
The project schedule and milestones are documented in an Excel Gantt chart.
You can open it directly from OneDrive:
[Open Gantt Chart (Excel)](https://1drv.ms/x/c/3c93911affd8d37b/ES31cw5MhRpEt13RNmkHWf4BVTB_VWwjtZepYwrf6UNFwQ?e=m4AKMq&nav=MTVfezAwMDAwMDAwLTAwMDEtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMH0)

## File Structure

src/</br>
├── model.py                 # CNN architecture for audio classification</br>
├── prepare_training_data.py # Model training pipeline</br>
├── main.py                  # Model testing and predictions</br>
├── test_model.py            # Performance evaluation</br>
└── save_test_samples.py     # Test data generation</br>
</br>
data/</br>
└── test/                    # Sample audio files for testing</br>

## How It Works
* Audio Processing: Converts WAV files to mel spectrograms

* Feature Extraction: Uses STFT and mel frequency scaling

* Classification: CNN model identifies language patterns

* Output: Predicts Finnish/English with confidence scores

## File Details

* Input: 16kHz WAV files (2-second segments)

* Output: Language prediction with accuracy metrics

* Model: 3-layer CNN with dropout regularization

* Training: ~10 minutes, ~85% accuracy on test data

**Test files should be placed in data/test/ with naming convention:**

* su_*.wav for Finnish samples

* en_*.wav for English samples


# Complete Project Setup and Usage Guide


## 1. Create Environment in Anaconda Prompt

- Open Anaconda Prompt from Windows Start menu

- Create new environment with Python 3.10</br>
conda create -n languagerecognition python=3.10

- Activate the environment</br>
conda activate languagerecognition


## 2. Install Required Dependencies

- Install TensorFlow using Conda (recommended for Windows)</br>
conda install tensorflow

- Install tensorflow-datasets</br>
conda install -c conda-forge tensorflow-datasets

- Install additional required libraries</br>
pip install soundfile numpy

- Alternative: If Conda installation fails, use pip
- pip install tensorflow-cpu tensorflow-datasets soundfile numpy

## 3. Verify Installation

- Test that all libraries work correctly

python -c "import tensorflow as tf; print('TensorFlow OK - Version:', tf.__version__)"</br>
python -c "import tensorflow_datasets as tfds; print('TensorFlow Datasets OK')"</br>
python -c "import soundfile; import numpy; print('SoundFile & NumPy OK')"</br>

## 4. Clone the Project from GitHub

- Navigate to your desired directory
- Clone the project</br>
git clone https://github.com/KippaK/tf-audio-languange-recognition-model.git

- Navigate to project directory</br>
cd tf-audio-languange-recognition-model

## 5. Project Usage

**Generate Test Data**

python src/save_test_samples.py</br>
*Creates 10 Finnish and 10 English audio samples in data/test/ directory*

**Train the model**

python src/prepare_training_data.py</br>
*Trains the CNN model (~10 minutes) and saves the best version as best_model.keras*

**Test the Model**

- Test with data/test/ audio files</br>
python src/main.py

- Or run comprehensive evaluation</br>
python src/test_model.py


# Expected Results

## Training Output Example

TRAINING RESULTS:</br>
Training accuracy: 0.9685</br>
Validation accuracy: 0.9525</br>
EXCELLENT performance!</br>

## Sample Predictions

en_001.wav | Prediction: English (95.3%) | CORRECT (True: English)</br>
en_002.wav | Prediction: English (100.0%) | CORRECT (True: English)</br>
en_003.wav | Prediction: English (67.8%) | CORRECT (True: English)</br>
su_001.wav | Prediction: Finnish (100.0%) | CORRECT (True: Finnish)</br>
su_002.wav | Prediction: Finnish (100.0%) | CORRECT (True: Finnish)</br>
su_003.wav | Prediction: English (63.7%) | WRONG (True: Finnish)</br>


SUMMARY:
Accuracy: 17/20 (85.0%)

Finnish accuracy: 10/10 (100.0%)</br>
English accuracy: 7/10 (70.0%)</br>

## Test Evaluation

TEST RESULTS:</br>
Test accuracy: 0.5850 (58.50%)</br>
Test loss: 1.3392</br>
ACCEPTABLE generalization</br>

## Performance Levels

* EXCELLENT: >75% validation accuracy
* GOOD: 65-75% validation accuracy
* ACCEPTABLE: 55-65% validation accuracy
* NEEDS IMPROVEMENT: <55% validation accuracy

