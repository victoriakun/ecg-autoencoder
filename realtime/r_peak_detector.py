"""Streaming R-peak detector built on a Pan-Tompkins-style pipeline.

Designed for the realtime windowing path: feed the raw ECG samples in
chunks, ask `pop_new_peaks()` periodically to harvest detected R-peak
indices (in absolute sample count). Subsequent calls only return peaks
that haven't been emitted before.

For thesis-grade demos we keep the implementation small and dependency-
free. It is good enough to align beat-centered windows on MIT-BIH at
360 Hz; for production deployment the obvious upgrade is wfdb.processing
.xqrs_detect or neurokit2.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class StreamingRPeakDetector:
    def __init__(
        self,
        fs: int = 360,
        bandpass_low: float = 5.0,
        bandpass_high: float = 15.0,
        refractory_sec: float = 0.25,
        integration_sec: float = 0.150,
    ) -> None:
        self._fs = fs
        self._refractory = int(refractory_sec * fs)
        self._integration_n = int(integration_sec * fs)
        self._sos = butter(
            2, [bandpass_low, bandpass_high], btype="band",
            fs=fs, output="sos",
        )
        self._zi = sosfilt_zi(self._sos)

        self._raw_total = 0
        self._integrated: deque = deque(maxlen=self._fs * 30)
        self._bandpassed: deque = deque(maxlen=self._fs * 30)
        self._integ_offset = 0
        self._peaks: list = []
        self._last_peak_abs = -10**9
        self._signal_level = 0.0
        self._noise_level = 0.0
        self._threshold = 0.0
        self._cursor = 0
        # The integration window introduces a lag of ~half its length
        self._search_back = self._integration_n // 2 + int(0.05 * fs)

    def push(self, samples: np.ndarray) -> None:
        """Feed a chunk of raw ECG samples (any length)."""
        x, self._zi = sosfilt(self._sos, samples, zi=self._zi)
        deriv = np.diff(x, prepend=x[0])
        squared = deriv ** 2
        kernel = np.ones(self._integration_n) / self._integration_n
        integ = np.convolve(squared, kernel, mode="same")
        for v in integ:
            self._integrated.append(v)
        for v in x:
            self._bandpassed.append(v)
        self._raw_total += samples.size
        if self._raw_total > len(self._integrated):
            self._integ_offset = self._raw_total - len(self._integrated)
        self._scan()

    def _scan(self) -> None:
        if len(self._integrated) < 2 * self._fs:
            return
        arr = np.asarray(self._integrated, dtype=float)
        # If we never set thresholds yet, initialise from first 2 s
        if self._threshold == 0.0:
            init = arr[: 2 * self._fs]
            self._signal_level = float(init.max() * 0.5)
            self._noise_level = float(init.mean())
            self._threshold = self._noise_level + 0.25 * (
                self._signal_level - self._noise_level
            )

        local_start = self._cursor - self._integ_offset
        local_start = max(local_start, 1)
        for local_i in range(local_start, len(arr) - 1):
            v = arr[local_i]
            abs_i = local_i + self._integ_offset
            if (
                v > arr[local_i - 1]
                and v > arr[local_i + 1]
                and v > self._threshold
                and abs_i - self._last_peak_abs > self._refractory
            ):
                refined = self._refine_peak(abs_i)
                self._peaks.append(refined)
                self._last_peak_abs = refined
                self._signal_level = 0.125 * v + 0.875 * self._signal_level
            else:
                self._noise_level = 0.125 * v + 0.875 * self._noise_level
            self._threshold = self._noise_level + 0.25 * (
                self._signal_level - self._noise_level
            )
        self._cursor = len(arr) + self._integ_offset

    def _refine_peak(self, integ_peak_abs: int) -> int:
        """Find the true R-peak in the bandpassed signal: search backwards
        from the integration-signal peak for the largest |amplitude|, since
        the integration filter delays the peak by ~half its width."""
        bp_offset = self._raw_total - len(self._bandpassed)
        local_end = integ_peak_abs - bp_offset
        local_start = max(0, local_end - self._search_back)
        local_end = min(len(self._bandpassed), local_end + 1)
        if local_end <= local_start:
            return integ_peak_abs
        bp_arr = np.asarray(self._bandpassed)
        window = bp_arr[local_start:local_end]
        local_max = int(np.argmax(np.abs(window)))
        return bp_offset + local_start + local_max

    def pop_new_peaks(self, since_abs_index: int) -> list:
        """Return all detected peaks with abs index >= `since_abs_index`."""
        return [p for p in self._peaks if p >= since_abs_index]
