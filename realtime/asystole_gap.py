"""Asystole / extreme-bradycardia gap detector.

A beat-centered autoencoder structurally cannot detect asystole — by
construction, no R-peak means no beat-window means no anomaly score. The
cardiologist explicitly listed *absent QRS* as priority-2 to catch. We
solve this with a tiny side detector that watches the stream of R-peak
timestamps coming out of ``StreamingRPeakDetector`` and fires when the
gap between consecutive peaks exceeds a clinical threshold:

    >= 3.0 s  -> asystole / pause   (clinical default; AHA pause >= 3 s)
    >= 2.0 s  -> severe bradycardia (HR <= 30 bpm, configurable)

Output is a sequence of ``GapEvent``s suitable for direct insertion into
the existing ``anomaly_events`` SQLite table (same schema, ``event_type``
distinguishes them from autoencoder anomalies).

This module is intentionally small and stateless beyond the last-seen peak.
It is independent of the autoencoder pipeline and runs in O(1) per peak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class GapEvent:
    """A clinically significant absence of R-peaks.

    ``severity`` is one of {"asystole", "severe_bradycardia"}.
    ``start_sample`` and ``end_sample`` are absolute sample indices in the
    same coordinate system as the R-peak detector's output.
    """
    severity: str
    gap_seconds: float
    start_sample: int
    end_sample: int


class AsystoleGapDetector:
    """Stateful watcher over R-peak timestamps.

    Usage::

        det = AsystoleGapDetector(fs=360)
        for peak_sample in r_peak_stream:
            for ev in det.observe(peak_sample):
                event_store.write(ev, ...)
        # On stream end, also flush the trailing window:
        for ev in det.flush(now_sample):
            event_store.write(ev, ...)
    """

    def __init__(
        self,
        fs: int = 360,
        asystole_seconds: float = 3.0,
        severe_brady_seconds: float = 2.0,
    ) -> None:
        if asystole_seconds < severe_brady_seconds:
            raise ValueError("asystole_seconds must be >= severe_brady_seconds")
        self._fs = fs
        self._asy_n = int(asystole_seconds * fs)
        self._brady_n = int(severe_brady_seconds * fs)
        self._last_peak: Optional[int] = None

    def observe(self, peak_sample: int) -> List[GapEvent]:
        """Register a newly-detected R-peak; return any gap-events the
        gap from the previous peak crossed."""
        events: List[GapEvent] = []
        if self._last_peak is not None:
            gap = peak_sample - self._last_peak
            ev = self._classify(self._last_peak, peak_sample, gap)
            if ev is not None:
                events.append(ev)
        self._last_peak = peak_sample
        return events

    def flush(self, now_sample: int) -> List[GapEvent]:
        """If the stream has been silent since the last peak, emit a gap
        event up to the current sample. Call on stream end or periodically
        from a watchdog (the gap event is *open-ended* — the caller should
        upgrade severity if a longer gap is observed later)."""
        if self._last_peak is None:
            return []
        gap = now_sample - self._last_peak
        ev = self._classify(self._last_peak, now_sample, gap)
        return [ev] if ev is not None else []

    def _classify(
        self, start: int, end: int, gap_n: int
    ) -> Optional[GapEvent]:
        if gap_n >= self._asy_n:
            return GapEvent("asystole", gap_n / self._fs, start, end)
        if gap_n >= self._brady_n:
            return GapEvent("severe_bradycardia", gap_n / self._fs, start, end)
        return None


def scan(
    peak_samples: Iterable[int],
    *,
    fs: int = 360,
    asystole_seconds: float = 3.0,
    severe_brady_seconds: float = 2.0,
    until_sample: Optional[int] = None,
) -> List[GapEvent]:
    """Convenience: scan a complete sequence of R-peaks offline."""
    det = AsystoleGapDetector(
        fs=fs,
        asystole_seconds=asystole_seconds,
        severe_brady_seconds=severe_brady_seconds,
    )
    out: List[GapEvent] = []
    last = 0
    for s in peak_samples:
        out.extend(det.observe(s))
        last = s
    if until_sample is not None and until_sample > last:
        out.extend(det.flush(until_sample))
    return out
