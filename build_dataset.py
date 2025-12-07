# build_dataset.py
import os
import wfdb
import numpy as np

from preprocess import preprocess  # our filtering + normalisation

# Folder where your downloaded MIT-BIH files are
DATA_DIR = "data/mitbih"

# Output file for the windowed dataset
OUTPUT_PATH = "data/mitbih_windows.npy"

# Subset of records to start with (you can add more later)
RECORDS = ["100", "101", "102", "103", "104"]

# Length of each window in seconds
WINDOW_SEC = 2.0


def load_signal(record_id: str):
    """
    Load one record from MIT-BIH and return the signal and sampling rate.
    """
    record_path = os.path.join(DATA_DIR, record_id)
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    # use lead 0 for now
    sig = record.p_signal[:, 0]  # shape: (N,)
    return sig, fs


def build_windows():
    all_windows = []

    for rec in RECORDS:
        print(f"Processing record {rec}")
        sig, fs = load_signal(rec)
        win_len = int(WINDOW_SEC * fs)  # samples per window

        # Number of full windows we can extract
        num_windows = len(sig) // win_len
        if num_windows == 0:
            print(f"Record {rec}: too short for one window, skipping.")
            continue

        # Trim the signal to a multiple of window length
        sig = sig[: num_windows * win_len]

        # Reshape into (num_windows, win_len)
        sig_reshaped = sig.reshape(num_windows, win_len)

        # Preprocess each window: band-pass + (optional notch) + normalization
        sig_pre = np.array([preprocess(w, fs) for w in sig_reshaped])

        print(f"  -> {num_windows} windows from record {rec}")
        all_windows.append(sig_pre)

    if not all_windows:
        print("No windows created. Check your RECORDS list and data folder.")
        return

    X = np.vstack(all_windows)  # shape: (total_windows, win_len)
    print("Final dataset shape:", X.shape)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, X)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_windows()
