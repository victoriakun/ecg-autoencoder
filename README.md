# ECG Anomaly Detection Using Convolutional Autoencoder

Unsupervised anomaly detection for ECG signals using a convolutional autoencoder trained on the MIT-BIH Arrhythmia Database.

## Overview

This project implements an autoencoder-based approach to detect cardiac arrhythmias in ECG recordings. The model is trained exclusively on normal heartbeats and learns to reconstruct typical ECG patterns. Anomalies (arrhythmias) are detected based on elevated reconstruction error.

## Results

Evaluated on the held-out MIT-BIH test split (16,407 windows: 13,594 normal, 2,813 anomalies):

| Metric | Value |
|---|---|
| ROC-AUC | **0.9750** |
| PR-AUC | 0.8455 |
| Sensitivity (Recall) | 0.9499 |
| Specificity | 0.9305 |
| Precision | 0.7387 |
| F1-Score | 0.8311 |

**Deployed operating point:** τ = 0.3096, chosen to yield 95% sensitivity on the held-out test split — favouring clinical sensitivity over precision. Full calibration record in `models/mitbih_autoencoder_best.calibration.json`.

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
├── realtime/                 # Real-time pipeline package (inference, UI, event store)
├── realtime_app.py           # Real-time pipeline entry point
├── tools/                    # Calibration, clinical proof, evaluation utilities
├── tests/                    # Unit and integration tests
└── docs/                     # Clinical proof artifacts and external evaluation results
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
- **Window Size**: 2 seconds (720 samples), beat-centered
- **Split**: 70% train, 15% validation, 15% test
- **Training**: Normal beats only (unsupervised learning)
- **Anomaly score**: p99 of per-sample squared residuals
- **Normalisation**: record-level z-score

## Real-time pipeline

The `realtime/` package adds a live pipeline that replays MIT-BIH records at
360 Hz, detects anomalies with dynamic thresholding and N-of-M smoothing, and
visualises 2–3 concurrent streams in a PyQt desktop UI.

### Headless demo (CI-friendly)

```bash
python realtime_app.py --headless --records 208 --seconds 30
```

### GUI mode

```bash
python realtime_app.py
```

### Configuration

Defaults live in `realtime/config_rt.py` (`RealtimeConfig` dataclass).
To override, write a JSON file and pass it with `--config`:

```bash
python realtime_app.py --config my_config.json
```

### Architecture

Each module in `realtime/` maps to a thesis Section 3.1.* subsection: signal ingest, scoring, dynamic thresholding, N-of-M smoothing, event store, and the PyQt UI.

## Clinical proof generation

Re-generate the cardiologist-blind PDFs used for external validation:

```bash
python tools/clinical_proof.py             # Anonymised blind set + answer key
python tools/clinical_examples_by_type.py  # Examples by arrhythmia type
```

Outputs land in `docs/clinical_proof/`.

## License

MIT License
