# ECG Anomaly Detection Using Convolutional Autoencoder

Unsupervised anomaly detection for ECG signals using a convolutional autoencoder trained on the MIT-BIH Arrhythmia Database.

## Overview

This project implements an autoencoder-based approach to detect cardiac arrhythmias in ECG recordings. The model is trained exclusively on normal heartbeats and learns to reconstruct typical ECG patterns. Anomalies (arrhythmias) are detected based on elevated reconstruction error.

## Results

- **ROC-AUC**: 97.21%
- **Accuracy**: 94.59%
- **Sensitivity**: 90.19%
- **Specificity**: 95.51%
- **F1-Score**: 85.14%

## Project Structure

```
ecg_ae_project/
├── config.py                 # Configuration parameters
├── preprocess.py             # Signal preprocessing (filtering, normalization)
├── models.py                 # Autoencoder architecture
├── dataset.py                # Dataset building with labels
├── train_mitbih.py           # Main training script
├── evaluate.py               # Evaluation metrics
├── visualize.py              # Visualization tools
└── evaluation_plots/         # Generated evaluation figures
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ecg-autoencoder.git
cd ecg-autoencoder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download MIT-BIH Data
```bash
python dataset.py --download
```

### 2. Build Dataset
```bash
python dataset.py
```

### 3. Train Model
```bash
python train_mitbih.py
```

### 4. Evaluate
```bash
python evaluate.py
```

## Model Architecture

The convolutional autoencoder consists of:
- **Encoder**: 4 Conv1D layers (720 → 32 dimensions)
- **Latent Space**: 32-dimensional compressed representation
- **Decoder**: 4 ConvTranspose1D layers (32 → 720 dimensions)
- **Total Parameters**: 583,361

## Data

- **Database**: MIT-BIH Arrhythmia Database (48 records, ~110,000 beats)
- **Sampling Rate**: 360 Hz
- **Window Size**: 2 seconds (720 samples)
- **Split**: 70% train, 15% validation, 15% test
- **Training**: Normal beats only (unsupervised learning)

## License

MIT License
