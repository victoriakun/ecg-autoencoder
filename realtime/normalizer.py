"""Per-patient streaming normalizer.

Mirrors the batch preprocessing methodology used in dataset.py / evaluate.py:
the entire record is normalized once with global zero-mean / unit-variance
statistics. We can't see the whole record while streaming, so we compute the
stats from a fixed-length warmup buffer of bandpass-filtered samples, then
freeze and apply them to every subsequent window.

This keeps real-time residuals on the SAME scale as the training/evaluation
pipeline, so the threshold stored in the checkpoint metrics is directly usable.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class WarmupNormalizer:
    def __init__(self, warmup_samples: int) -> None:
        if warmup_samples <= 0:
            raise ValueError("warmup_samples must be positive")
        self._target = warmup_samples
        self._buffer: list = []
        self._collected = 0
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._mean is not None

    @property
    def stats(self) -> tuple[Optional[float], Optional[float]]:
        return (self._mean, self._std)

    def observe(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Accept a bandpassed window. Return normalized window once warmup
        is complete; return None until then."""
        with self._lock:
            if self._mean is not None:
                return (samples - self._mean) / self._std

            self._buffer.append(samples.astype(float))
            self._collected += samples.size
            if self._collected < self._target:
                return None

            concat = np.concatenate(self._buffer)[:self._target]
            self._mean = float(concat.mean())
            self._std = float(concat.std()) + 1e-8
            self._buffer = []
            return (samples - self._mean) / self._std
