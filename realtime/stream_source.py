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
    record_offset_samples: int = 0  # sample index of window centre in the record


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
        windowing_mode: str = "stride",
    ) -> None:
        """`windowing_mode`:
          - "stride"        cut a window every `stride_samples` (default)
          - "beat_centered" run streaming R-peak detection and emit one
                            window centred on each detected R-peak
        """
        self._patient_id = patient_id
        self._record_path = str(record_path)
        self._window = window_samples
        self._stride = stride_samples
        self._clock = clock
        self._out = out_queue
        self._stop = stop_event
        self._mode = windowing_mode

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

        if self._mode == "beat_centered":
            self._run_beat_centered(signal, fs)
        else:
            self._run_stride(signal, fs)

    def _run_stride(self, signal, fs):
        stride_seconds = self._stride / float(fs)
        start = 0
        while not self._stop.is_set():
            end = start + self._window
            if end > signal.size:
                break
            window = signal[start:end].copy()
            centre = start + self._window // 2
            self._emit(window, centre)
            start += self._stride
            self._clock.sleep(stride_seconds)

    def _run_beat_centered(self, signal, fs):
        """Pre-compute R-peaks on the full loaded signal with wfdb's
        validated XQRS detector, then emit one beat-centered window per
        peak at wall-clock pace. This is appropriate for the file-replay
        demo; a true online deployment would use the streaming detector
        in realtime.r_peak_detector."""
        try:
            from wfdb.processing import XQRS
            xqrs = XQRS(sig=signal, fs=int(fs))
            xqrs.detect(verbose=False)
            peaks = list(xqrs.qrs_inds)
        except Exception as e:
            log.error("XQRS failed for %s, falling back to streaming detector: %s",
                      self._patient_id, e)
            from realtime.r_peak_detector import StreamingRPeakDetector
            det = StreamingRPeakDetector(fs=int(fs))
            for i in range(0, signal.size, self._stride):
                det.push(signal[i: i + self._stride])
            peaks = det.pop_new_peaks(0)

        half = self._window // 2
        chunk_seconds = self._stride / float(fs)
        cursor = 0
        peak_iter = iter(peaks)
        next_peak = next(peak_iter, None)
        while not self._stop.is_set() and cursor < signal.size:
            cursor = min(cursor + self._stride, signal.size)
            while next_peak is not None and next_peak + half <= cursor:
                if next_peak - half >= 0 and next_peak + half <= signal.size:
                    window = signal[next_peak - half: next_peak + half].copy()
                    self._emit(window, next_peak)
                next_peak = next(peak_iter, None)
            self._clock.sleep(chunk_seconds)

    def _emit(self, window, anchor_idx):
        if np.isnan(window).any():
            log.warning("NaN in window at offset %d (%s); skipping",
                        anchor_idx, self._patient_id)
            return
        try:
            self._out.put(
                StreamWindow(
                    patient_id=self._patient_id,
                    ts_utc=_now_iso(),
                    samples=window,
                    record_offset_samples=int(anchor_idx),
                ),
                timeout=1.0,
            )
        except queue.Full:
            log.warning("queue full for %s; dropping window",
                        self._patient_id)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
