"""Per-stream UI panel: raw + recon + residual + anomaly highlights."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from realtime.detector import DetectionResult
from realtime.inference import InferenceResult
from realtime.notifier import AnomalyEdge


BUFFER_SECONDS = 10
RED_BADGE_HOLD_MS = 3000  # min ms to keep red badge after a rising edge


@dataclass
class _RingBuffer:
    capacity: int
    data: np.ndarray

    @classmethod
    def create(cls, capacity: int) -> "_RingBuffer":
        return cls(capacity=capacity, data=np.zeros(capacity, dtype=float))

    def push_chunk(self, chunk: np.ndarray) -> None:
        n = chunk.size
        if n >= self.capacity:
            self.data = chunk[-self.capacity:].copy()
        else:
            self.data = np.concatenate([self.data[n:], chunk])


def _fmt_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class StreamPanel(QWidget):
    def __init__(self, patient_id: str, sampling_rate: int,
                 stride_samples: int, parent=None) -> None:
        super().__init__(parent)
        self._patient_id = patient_id
        self._fs = sampling_rate
        self._cap = BUFFER_SECONDS * sampling_rate
        self._stride = stride_samples
        self._raw = _RingBuffer.create(self._cap)
        self._recon = _RingBuffer.create(self._cap)
        self._resid: deque = deque(
            maxlen=BUFFER_SECONDS * (sampling_rate // stride_samples)
        )
        self._threshold_val = None
        self._x = np.arange(self._cap, dtype=float) / sampling_rate
        self._latest_offset_sec = 0.0
        self._red_badge_until = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel(
            f"Record {self._patient_id} — single-lead ECG (MLII)"
        )
        title.setStyleSheet("font-weight: bold; font-size: 13pt;")
        header.addWidget(title)
        self._clock_label = QLabel("t = --:--")
        self._clock_label.setStyleSheet(
            "color: #555; font-family: monospace; font-size: 11pt;"
        )
        header.addSpacing(12)
        header.addWidget(self._clock_label)
        header.addStretch()
        self._status = QLabel("● WARMUP")
        f = QFont(); f.setPointSize(14); f.setBold(True)
        self._status.setFont(f)
        self._status.setStyleSheet("color: grey; padding: 4px 10px;")
        header.addWidget(self._status)
        layout.addLayout(header)

        explanation = QLabel(
            "<b>Top plot:</b> the most recent 10 s of streamed ECG. "
            "Blue = preprocessed signal (band-pass + global z-score, so "
            "y-axis is in standard deviations). "
            "Orange = the autoencoder's reconstruction. "
            "<b>Bottom plot:</b> reconstruction error (residual) of every "
            "2-s window vs the dashed red threshold. An "
            "alert fires only when 2 of the last 3 windows exceed the threshold."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #444; font-size: 10pt;")
        layout.addWidget(explanation)

        self._alert_banner = QLabel("")
        self._alert_banner.setAlignment(Qt.AlignCenter)
        self._alert_banner.setFont(f)
        self._alert_banner.setStyleSheet(
            "background: transparent; color: transparent; padding: 6px;"
        )
        self._alert_banner.setFixedHeight(34)
        layout.addWidget(self._alert_banner)

        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._clear_banner)

        self._raw_plot = pg.PlotWidget(title="ECG (z-scored): blue = original, orange = model reconstruction")
        self._raw_plot.setLabel("left", "amplitude (σ, z-scored)")
        self._raw_plot.setLabel("bottom", "time within visible window (s)")
        self._raw_curve = self._raw_plot.plot(pen=pg.mkPen("#1f77b4"))
        self._recon_curve = self._raw_plot.plot(pen=pg.mkPen("#ff7f0e"))
        self._anomaly_regions: list = []
        layout.addWidget(self._raw_plot)

        self._res_plot = pg.PlotWidget(
            title="Reconstruction error per 2-s window (one point every 0.5 s)"
        )
        self._res_plot.setLabel("left", "MSE between original and reconstruction")
        self._res_plot.setLabel("bottom", "window index (most recent on the right)")
        self._res_curve = self._res_plot.plot(pen=pg.mkPen("#555555"))
        self._thr_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("#d62728", style=Qt.DashLine)
        )
        self._res_plot.addItem(self._thr_line)
        layout.addWidget(self._res_plot)

    def _clear_banner(self) -> None:
        self._alert_banner.setText("")
        self._alert_banner.setStyleSheet(
            "background: transparent; color: transparent; padding: 6px;"
        )

    def on_window(self, r: InferenceResult) -> None:
        if r.patient_id != self._patient_id:
            return
        chunk = r.raw[-self._stride:] if r.raw.size >= self._stride else r.raw
        recon_chunk = r.recon[-self._stride:] if r.recon.size >= self._stride else r.recon
        self._raw.push_chunk(chunk)
        self._recon.push_chunk(recon_chunk)
        self._raw_curve.setData(self._x, self._raw.data)
        self._recon_curve.setData(self._x, self._recon.data)
        if hasattr(r, "record_offset_samples"):
            self._latest_offset_sec = r.record_offset_samples / float(self._fs)
            self._clock_label.setText(
                f"t = {_fmt_mmss(self._latest_offset_sec)}  "
                f"(sample {r.record_offset_samples})"
            )

    def on_detection(self, d: DetectionResult) -> None:
        if d.patient_id != self._patient_id:
            return
        self._resid.append(d.residual)
        self._res_curve.setData(np.arange(len(self._resid)), np.asarray(self._resid))
        if d.threshold is not None:
            self._threshold_val = d.threshold
            self._thr_line.setValue(d.threshold)

        # Hold the red badge for at least RED_BADGE_HOLD_MS after a rising
        # edge, so even a 1-window anomaly stays clearly visible.
        now_ms = time.monotonic() * 1000
        if d.state == "anomaly":
            label, color = "ANOMALY", "red"
            self._red_badge_until = now_ms + RED_BADGE_HOLD_MS
        elif now_ms < self._red_badge_until:
            label, color = "ANOMALY (cooldown)", "red"
        elif d.state == "warmup":
            label, color = "WARMUP", "grey"
        elif d.exceeded:
            label, color = "WATCHING", "#f0a020"
        else:
            label, color = "NORMAL", "green"
        self._status.setText(f"● {label}")
        self._status.setStyleSheet(
            f"color: white; background: {color}; padding: 4px 10px; border-radius: 4px;"
        )

    def on_edge(self, edge: AnomalyEdge) -> None:
        if edge.patient_id != self._patient_id:
            return
        if edge.event_type == "anomaly_start":
            right = self._x[-1]
            region = pg.LinearRegionItem(
                values=(right - 0.5, right),
                brush=QColor(214, 39, 40, 80),
                movable=False,
            )
            self._raw_plot.addItem(region)
            self._anomaly_regions.append(region)
            ts = edge.ts_utc.split("T")[-1][:12] if "T" in edge.ts_utc else edge.ts_utc
            self._alert_banner.setText(
                f"⚠  ANOMALY DETECTED  rec {edge.patient_id} "
                f"t = {_fmt_mmss(self._latest_offset_sec)}   "
                f"residual={edge.residual:.3f} > threshold={edge.threshold:.3f}"
            )
            self._alert_banner.setStyleSheet(
                "background: #d62728; color: white; padding: 6px; border-radius: 4px;"
            )
            self._banner_timer.start(5000)
