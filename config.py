# config.py
"""Centralized configuration for ECG anomaly detection project."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MITBIH_DIR = DATA_DIR / "mitbih"
WINDOWS_PATH = DATA_DIR / "mitbih_windows.npy"
LABELS_PATH = DATA_DIR / "mitbih_labels.npy"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "ecg_autoencoder.pt"

# MIT-BIH database - all 48 records
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219",
    "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]

# Normal beat symbols (used for training)
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

# Anomalous beat symbols
ANOMALY_SYMBOLS = {
    "A", "a", "J", "S",  # Supraventricular
    "V", "E",            # Ventricular
    "F",                 # Fusion
    "/", "f", "Q",       # Paced / Unknown
}

# Signal processing
SAMPLING_RATE = 360  # Hz (MIT-BIH default)
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)  # 720 samples

# Preprocessing
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 40.0  # Hz
FILTER_ORDER = 4

# Training
SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model architecture
LATENT_DIM = 32

# Early stopping
PATIENCE = 10
MIN_DELTA = 1e-6
