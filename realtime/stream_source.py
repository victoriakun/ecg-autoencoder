"""Replays a saved ECG record into a queue at wall-clock pace."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# Indirection so tests can monkeypatch
def _rdrecord(path: str):
    import wfdb
    return wfdb.rdrecord(path)


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class StreamWindow:
    patient_id: str
    ts_utc: str
    samples: np.ndarray  # shape (window_samples,)


class StreamSource:
    def __init__(
        self,
        patient_id: str,
        record_path: str | Path,
        window_samples: int,
        stride_samples: int,
        clock: Clock,
        out_queue: "queue.Queue[StreamWindow]",
        stop_event: threading.Event,
    ) -> None:
        self._patient_id = patient_id
        self._record_path = str(record_path)
        self._window = window_samples
        self._stride = stride_samples
        self._clock = clock
        self._out = out_queue
        self._stop = stop_event

    def run(self) -> None:
        try:
            rec = _rdrecord(self._record_path)
        except Exception as e:
            log.error("failed to load record %s: %s", self._record_path, e)
            return

        signal = np.asarray(rec.p_signal[:, 0], dtype=float)
        fs = getattr(rec, "fs", 360)
        if signal.size < self._window:
            log.warning(
                "record %s too short (%d samples < %d); skipping",
                self._record_path, signal.size, self._window,
            )
            return

        stride_seconds = self._stride / float(fs)
        start = 0
        while not self._stop.is_set():
            end = start + self._window
            if end > signal.size:
                break
            window = signal[start:end].copy()
            if np.isnan(window).any():
                log.warning("NaN in window at offset %d (%s); skipping",
                            start, self._patient_id)
            else:
                try:
                    self._out.put(
                        StreamWindow(
                            patient_id=self._patient_id,
                            ts_utc=_now_iso(),
                            samples=window,
                        ),
                        timeout=1.0,
                    )
                except queue.Full:
                    log.warning("queue full for %s; dropping window",
                                self._patient_id)
            start += self._stride
            self._clock.sleep(stride_seconds)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
