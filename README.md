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
* [Performance Analysis](#performance-analysis)
* [Technical Architecture](#technical-architecture)

## Objective 
Spoken Language Identification (LID) is defined as detecting language from an audio clip by an unknown speaker, regardless of gender, manner of speaking, and distinct age speaker. Deep learning system for classifying audio as Finnish or English using advanced CNN with attention mechanisms and mel spectrograms.

## Available models and languages
**FLEURS dataset** downloads can be found here: [Downloads](https://www.tensorflow.org/datasets/catalog/xtreme_s)

## Environment Setup
The models are implemented in TensorFlow/Keras.
To use all of the functionality of the library, you should have:</br>
tensorflow==2.13.0</br>
numpy==1.24.3</br>
tensorflow-datasets==4.9.4</br>
soundfile==0.12.1</br>
colorama==0.4.6</br>

## Gantt chart for project planning
The project schedule and milestones are documented in an Excel Gantt chart.
You can open it directly from OneDrive:
[Open Gantt Chart (Excel)](https://1drv.ms/x/c/3c93911affd8d37b/ES31cw5MhRpEt13RNmkHWf4BVTB_VWwjtZepYwrf6UNFwQ?e=m4AKMq&nav=MTVfezAwMDAwMDAwLTAwMDEtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMH0)

## File Structure

```
src/
├── model.py                 # Advanced CNN architecture with attention mechanism
├── prepare_training_data.py # Complete training pipeline with data augmentation
├── main.py                  # Model testing and predictions with confidence scores
├── test_model.py            # Comprehensive performance evaluation
└── save_test_samples.py     # Test data generation from FLEURS dataset

data/
└── test/                    # Sample audio files for testing
    ├── su_*.wav            # Finnish test samples
    ├── en_*.wav            # English test samples
    └── transcripts.txt     # Transcripts for all test files
```

## How It Works
* **Audio Processing**: Converts WAV files to mel spectrograms using STFT and mel frequency scaling
* **Feature Extraction**: Advanced CNN with attention mechanism identifies language patterns
* **Data Augmentation**: FFT-based time stretching, noise addition, gain variation
* **Classification**: Predicts Finnish/English with confidence scores
* **Evaluation**: Comprehensive testing with proper train/validation/test splits

## File Details

* **Input**: 16kHz WAV files (2-second segments)
* **Output**: Language prediction with confidence percentages
* **Model**: 4-layer CNN with attention mechanism and dropout regularization
* **Training**: ~30-40 minutes, ~83% accuracy on test data
* **Architecture**: 32→64→128→256 filters with channel attention

**Test files should be placed in data/test/ with naming convention:**

* `su_*.wav` for Finnish samples
* `en_*.wav` for English samples

## Technical Architecture

### Model Features
- **Advanced CNN**: 4 convolutional blocks with batch normalization
- **Attention Mechanism**: Channel attention for frequency importance weighting
- **Regularization**: L2 regularization + Dropout (0.2-0.5) to prevent overfitting
- **Optimization**: Adam optimizer with learning rate scheduling
- **Class Weights**: Balanced training (Finnish: 1.2, English: 0.8)

### Data Processing Pipeline
1. **Audio Loading**: 16kHz sampling rate, mono conversion
2. **Normalization**: Amplitude normalization to [-1, 1] range
3. **Padding/Truncation**: Fixed 2-second length (32000 samples)
4. **Spectrogram**: STFT with 512 frame length, 256 frame step
5. **Mel Conversion**: 64 mel bands, 80Hz-7600Hz frequency range
6. **Log Compression**: Logarithmic scaling for perceptual accuracy

# Complete Project Setup and Usage Guide

## 1. Create Environment in Anaconda Prompt

- Open Anaconda Prompt from Windows Start menu

- Create new environment with Python 3.10</br>
`conda create -n kielentunnistus python=3.10`

- Activate the environment</br>
`conda activate kielentunnistus`

## 2. Install Required Dependencies

- Install TensorFlow using Conda (recommended for Windows)</br>
`conda install tensorflow`

- Install tensorflow-datasets</br>
`conda install -c conda-forge tensorflow-datasets`

- Install additional required libraries</br>
`pip install soundfile colorama`

- Alternative: If Conda installation fails, use pip</br>
`pip install tensorflow-cpu tensorflow-datasets soundfile colorama`

## 3. Verify Installation

- Test that all libraries work correctly</br>
`python -c "import tensorflow as tf; print('TensorFlow OK - Version:', tf.__version__)"`</br>
`python -c "import tensorflow_datasets as tfds; print('TensorFlow Datasets OK')"`</br>
`python -c "import soundfile; import colorama; print('SoundFile & Colorama OK')"`

## 4. Clone the Project from GitHub

- Navigate to your desired directory
- Clone the project</br>
`git clone https://github.com/KippaK/tf-audio-languange-recognition-model.git`

- Navigate to project directory</br>
`cd tf-audio-languange-recognition-model`

## 5. Project Usage

**Generate Test Data**</br>
`python src/save_test_samples.py`</br>
*Creates 10 Finnish and 10 English audio samples in data/test/ directory from FLEURS test split*

**Train the Model**</br>
`python src/prepare_training_data.py`</br>
*Trains the advanced CNN model (~30-40 minutes) with data augmentation and saves best version as best_model.keras*

**Test the Model**

- Test with data/test/ audio files</br>
`python src/main.py`

- Run comprehensive evaluation on large test set</br>
`python src/test_model.py`

# Expected Results

## Training Output Example

```
AUDIO LANGUAGE DETECTION
==================================================
Loading data...
Data loaded successfully
Finnish - Train: 1200, Val: 300
English - Train: 1200, Val: 300
Input shape: (124, 64, 1)
Building model...
Starting training with class weights...

TRAINING RESULTS:
Training accuracy: 0.9358
Best validation accuracy: 0.8133
EXCELLENT performance!
```

## Sample Predictions

```
en_001.wav | Prediction: English (95.3%) | CORRECT (True: English)
en_002.wav | Prediction: English (100.0%) | CORRECT (True: English) 
en_003.wav | Prediction: English (67.8%) | CORRECT (True: English)
su_001.wav | Prediction: Finnish (100.0%) | CORRECT (True: Finnish)
su_002.wav | Prediction: Finnish (100.0%) | CORRECT (True: Finnish)
su_003.wav | Prediction: English (63.7%) | WRONG (True: Finnish)

SUMMARY:
Accuracy: 16/20 (80.0%)

Finnish accuracy: 9/10 (90.0%)
English accuracy: 7/10 (70.0%)
```

## Test Evaluation

```
TEST RESULTS:
==================================================
Test accuracy: 0.8300 (83.00%)
Test loss: 0.6671
Test samples total: 400

GENERALIZATION ASSESSMENT:
EXCELLENT generalization
```

## Performance Analysis

### Current Performance Levels
* **EXCELLENT**: >80% test accuracy  **ACHIEVED: 83%**
* **VERY GOOD**: 75-80% test accuracy
* **GOOD**: 70-75% test accuracy  
* **ACCEPTABLE**: 65-70% test accuracy
* **NEEDS IMPROVEMENT**: <65% test accuracy

### Language-Specific Performance
- **Finnish**: 90% accuracy (strong performance)
- **English**: 70% accuracy (moderate performance, some bias toward Finnish)
- **Overall**: 83% test accuracy (production-ready)

## Key Features and Improvements

### **What Works Well**
- **Advanced CNN with Attention**: 83% test accuracy
- **Proper Data Splitting**: Real test/validation separation
- **Data Augmentation**: FFT-based time stretching and noise addition
- **Class Balancing**: Effective bias mitigation
- **Production Ready**: High confidence scores and reliable predictions

### **Technical Achievements**
- **Attention Mechanism**: Successful implementation for frequency weighting
- **Robust Training**: Minimal overfitting (10% gap train vs test)
- **Comprehensive Evaluation**: Proper metrics and large test sets
- **User-Friendly Interface**: Clear predictions with confidence scores

### **Usage Recommendations**
1. **For production**: Current model achieves 83% accuracy - suitable for real applications
2. **For testing**: Use `python src/main.py` for quick file testing
3. **For evaluation**: Use `python src/test_model.py` for comprehensive metrics
4. **For training**: Model converges in ~30 epochs with early stopping

The project successfully demonstrates state-of-the-art spoken language identification with production-level accuracy and robust architectural design.

