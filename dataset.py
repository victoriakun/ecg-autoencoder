# dataset.py
"""Dataset building with proper labeling for ECG anomaly detection."""
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import wfdb

from config import (
    MITBIH_DIR, WINDOWS_PATH, LABELS_PATH, MITBIH_RECORDS,
    SAMPLING_RATE, WINDOW_SAMPLES, NORMAL_SYMBOLS, ANOMALY_SYMBOLS,
)
from preprocess import preprocess


def download_all_records() -> None:
    """Download all MIT-BIH records."""
    MITBIH_DIR.mkdir(parents=True, exist_ok=True)

    for rec in MITBIH_RECORDS:
        rec_path = MITBIH_DIR / f"{rec}.dat"
        if rec_path.exists():
            print(f"Record {rec} already exists, skipping...")
            continue

        print(f"Downloading record {rec}...")
        try:
            wfdb.dl_database("mitdb", dl_dir=str(MITBIH_DIR), records=[rec])
        except Exception as e:
            print(f"  Warning: Failed to download {rec}: {e}")

    print("Download complete.")


def get_beat_windows(
    record_id: str,
    window_samples: int = WINDOW_SAMPLES,
    lead: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract beat-centered windows with labels.

    Instead of sliding windows, we center each window around annotated beats.
    This gives us ground truth labels for anomaly detection.

    Args:
        record_id: MIT-BIH record ID
        window_samples: Number of samples per window
        lead: Which ECG lead to use (0 or 1)

    Returns:
        windows: Array of shape (n_beats, window_samples)
        labels: Array of 0 (normal) or 1 (anomaly)
        symbols: Original annotation symbols
    """
    record_path = str(MITBIH_DIR / record_id)

    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        return np.array([]), np.array([]), []

    signal = record.p_signal[:, lead]
    fs = record.fs

    # Preprocess entire signal first
    signal = preprocess(signal, fs, apply_bandpass=True, apply_notch=False)

    half_win = window_samples // 2
    windows = []
    labels = []
    symbols = []

    for idx, symbol in zip(annotation.sample, annotation.symbol):
        # Skip if window would go out of bounds
        if idx - half_win < 0 or idx + half_win > len(signal):
            continue

        # Skip non-beat annotations (rhythm changes, etc.)
        if symbol not in NORMAL_SYMBOLS and symbol not in ANOMALY_SYMBOLS:
            continue

        window = signal[idx - half_win : idx + half_win]

        # Ensure correct length (edge case handling)
        if len(window) != window_samples:
            continue

        windows.append(window)
        symbols.append(symbol)

        # Label: 0 = normal, 1 = anomaly
        if symbol in NORMAL_SYMBOLS:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64), symbols


def build_dataset(records: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build complete dataset from MIT-BIH records.

    Args:
        records: List of record IDs to use (None = all)

    Returns:
        X: Windows array (N, window_samples)
        y: Labels array (N,) with 0=normal, 1=anomaly
    """
    if records is None:
        records = MITBIH_RECORDS

    all_windows = []
    all_labels = []
    symbol_counts: Dict[str, int] = {}

    for rec in records:
        print(f"Processing record {rec}...")
        windows, labels, symbols = get_beat_windows(rec)

        if len(windows) == 0:
            print(f"  -> No valid windows, skipping")
            continue

        all_windows.append(windows)
        all_labels.append(labels)

        # Count symbols
        for s in symbols:
            symbol_counts[s] = symbol_counts.get(s, 0) + 1

        n_normal = np.sum(labels == 0)
        n_anomaly = np.sum(labels == 1)
        print(f"  -> {len(windows)} beats ({n_normal} normal, {n_anomaly} anomaly)")

    if not all_windows:
        raise ValueError("No windows created. Check your records and data folder.")

    X = np.vstack(all_windows)
    y = np.concatenate(all_labels)

    print(f"\nDataset summary:")
    print(f"  Total beats: {len(X)}")
    print(f"  Normal: {np.sum(y == 0)} ({100 * np.mean(y == 0):.1f}%)")
    print(f"  Anomaly: {np.sum(y == 1)} ({100 * np.mean(y == 1):.1f}%)")
    print(f"\nSymbol distribution:")
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
        label = "normal" if symbol in NORMAL_SYMBOLS else "anomaly"
        print(f"  {symbol}: {count} ({label})")

    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Save dataset to disk."""
    WINDOWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(WINDOWS_PATH, X)
    np.save(LABELS_PATH, y)
    print(f"\nSaved:")
    print(f"  Windows: {WINDOWS_PATH}")
    print(f"  Labels: {LABELS_PATH}")


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from disk."""
    X = np.load(WINDOWS_PATH)
    y = np.load(LABELS_PATH)
    return X, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ECG dataset")
    parser.add_argument("--download", action="store_true", help="Download MIT-BIH records first")
    parser.add_argument("--records", type=str, nargs="+", default=None, help="Specific records to use")
    args = parser.parse_args()

    if args.download:
        download_all_records()

    X, y = build_dataset(args.records)
    save_dataset(X, y)
