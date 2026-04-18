"""Dynamic threshold strategies for anomaly detection."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np


class Threshold(Protocol):
    def observe(self, residual: float) -> None: ...
    def current(self) -> float | None: ...


class _RollingBase:
    def __init__(self, w: int, warmup_min: int) -> None:
        if warmup_min > w:
            raise ValueError("warmup_min must be <= w")
        self._w = w
        self._warmup_min = warmup_min
        self._buf: deque[float] = deque(maxlen=w)

    def observe(self, residual: float) -> None:
        self._buf.append(float(residual))

    def _ready(self) -> bool:
        return len(self._buf) >= self._warmup_min


class PercentileThreshold(_RollingBase):
    def __init__(self, q: float, w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        self._q = q

    def current(self) -> float | None:
        if not self._ready():
            return None
        return float(np.quantile(self._buf, self._q))


class ZScoreThreshold(_RollingBase):
    def __init__(self, k: float, w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        self._k = k

    def current(self) -> float | None:
        if not self._ready():
            return None
        arr = np.asarray(self._buf, dtype=float)
        return float(arr.mean() + self._k * arr.std())


class FixedOnlineThreshold(_RollingBase):
    """Start from calibrated p99, nudge by +/-nudge based on recent median."""

    def __init__(self, calibration_path: Path, nudge: float,
                 w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        payload = json.loads(Path(calibration_path).read_text())
        self._base = float(payload["p99"])
        self._nudge = nudge

    def current(self) -> float | None:
        if not self._ready():
            return self._base
        median = float(np.median(self._buf))
        ratio = median / self._base if self._base else 0.0
        # If recent median is low relative to base, reduce threshold
        if ratio < 1.0:
            return self._base * (1.0 - self._nudge)
        if ratio > 1.0:
            return self._base * (1.0 + self._nudge)
        return self._base


def make_threshold(mode: str, *, w: int, warmup_min: int, **kwargs) -> Threshold:
    if mode == "percentile":
        return PercentileThreshold(q=kwargs["q"], w=w, warmup_min=warmup_min)
    if mode == "zscore":
        return ZScoreThreshold(k=kwargs["k"], w=w, warmup_min=warmup_min)
    if mode == "fixed_online":
        return FixedOnlineThreshold(
            calibration_path=kwargs["calibration_path"],
            nudge=kwargs["nudge"],
            w=w, warmup_min=warmup_min,
        )
    raise ValueError(f"unknown threshold mode: {mode}")
