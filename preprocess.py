# preprocess.py
"""Signal preprocessing functions for ECG data."""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Band-pass filter for ECG signals.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal.
    fs : float
        Sampling frequency in Hz.
    low : float
        Low cut-off frequency in Hz.
    high : float
        High cut-off frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    b, a = butter(order, [low_norm, high_norm], btype="band")
    # filtfilt = zero-phase (no phase distortion)
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray,
                 fs: float,
                 freq: float = 50.0,
                 quality: float = 30.0) -> np.ndarray:
    """
    Notch filter to remove powerline interference at 50 or 60 Hz.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal.
    fs : float
        Sampling frequency in Hz.
    freq : float
        Notch frequency (50 or 60 Hz).
    quality : float
        Quality factor: higher = narrower notch.

    Returns
    -------
    np.ndarray
        Notch-filtered ECG signal.
    """
    w0 = freq / (fs / 2.0)  # Normalized frequency
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization per window.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal window.

    Returns
    -------
    np.ndarray
        Normalized signal with ~0 mean and ~1 std.
    """
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8  # avoid division by zero
    return (signal - mean) / std


def preprocess(signal: np.ndarray,
               fs: float,
               apply_bandpass: bool = True,
               apply_notch: bool = False,
               notch_freq: float = 50.0) -> np.ndarray:
    """
    Full preprocessing pipeline used before feeding data to the autoencoder.

    Steps:
    1. Optional band-pass filtering (default 0.5–40 Hz).
    2. Optional notch filtering at 50/60 Hz.
    3. Normalization (zero-mean, unit-variance).

    Parameters
    ----------
    signal : np.ndarray
        1D raw ECG signal or window.
    fs : float
        Sampling frequency in Hz.
    apply_bandpass : bool
        Whether to apply band-pass filtering.
    apply_notch : bool
        Whether to apply powerline notch filtering.
    notch_freq : float
        Powerline frequency (50 or 60 Hz).

    Returns
    -------
    np.ndarray
        Preprocessed ECG signal (ready for autoencoder).
    """
    x = np.asarray(signal, dtype=float)

    if apply_bandpass:
        x = bandpass_filter(x, fs)

    if apply_notch:
        x = notch_filter(x, fs, freq=notch_freq)

    x = normalize(x)
    return x


if __name__ == "__main__":
    # quick self-test on random noise
    fs_test = 360.0
    t = np.linspace(0, 2.0, int(2.0 * fs_test), endpoint=False)
    test_sig = 0.5 * np.sin(2 * np.pi * 1.0 * t) + \
        0.1 * np.random.randn(len(t))

    x_pre = preprocess(test_sig, fs_test)
    print("Preprocessed signal: mean ≈", x_pre.mean(), "std ≈", x_pre.std())
