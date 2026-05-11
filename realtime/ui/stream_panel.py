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
MAX_ANOMALY_REGIONS = 6   # most recent N rising-edge highlights kept on chart


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
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel(
            f"Record {self._patient_id} — single-lead ECG (MLII)"
        )
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header.addWidget(title)
        self._clock_label = QLabel("t = --:--")
        self._clock_label.setStyleSheet(
            "color: #555; font-family: monospace; font-size: 10pt;"
        )
        header.addSpacing(10)
        header.addWidget(self._clock_label)
        header.addStretch()
        self._status = QLabel("● WARMUP")
        f = QFont(); f.setPointSize(12); f.setBold(True)
        self._status.setFont(f)
        self._status.setStyleSheet("color: grey; padding: 2px 8px;")
        # Tooltip explains the four states the cardiologist will see.
        self._status.setToolTip(
            "<b>Status</b>:<br>"
            "&bull; <b>WARMUP</b> — collecting baseline statistics, no decisions yet.<br>"
            "&bull; <b>NORMAL</b> — reconstruction error below threshold.<br>"
            "&bull; <b>WATCHING</b> — one window over threshold; waiting for "
            "confirmation (2 of last 3 windows must exceed before alerting).<br>"
            "&bull; <b>ANOMALY</b> — alert fired and being held for "
            f"{RED_BADGE_HOLD_MS//1000} s after the rising edge."
        )
        header.addWidget(self._status)
        layout.addLayout(header)

        # Compact one-line explanation; full version is in the README.
        explanation = QLabel(
            "Blue = ECG (z-scored) · orange = AE reconstruction · "
            "bottom = p99 score vs dashed-red threshold (alert when 2 of last 3 windows exceed)."
        )
        explanation.setWordWrap(False)
        explanation.setStyleSheet("color: #555; font-size: 9pt;")
        layout.addWidget(explanation)

        self._alert_banner = QLabel("")
        self._alert_banner.setAlignment(Qt.AlignCenter)
        ab_font = QFont(); ab_font.setPointSize(11); ab_font.setBold(True)
        self._alert_banner.setFont(ab_font)
        self._alert_banner.setStyleSheet(
            "background: transparent; color: transparent; padding: 2px;"
        )
        self._alert_banner.setMaximumHeight(24)
        self._alert_banner.setMinimumHeight(0)
        layout.addWidget(self._alert_banner)

        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._clear_banner)

        # Constrain plot heights so both ECG and residual fit on a typical
        # 800-px-tall window without scrolling.
        AXIS_PEN = pg.mkPen("#222222", width=1)
        TEXT_PEN = pg.mkPen("#222222")

        def _style_plot(plot: pg.PlotWidget) -> None:
            """White background + dark axes — readable in screenshots."""
            plot.setBackground("w")
            for axis_name in ("left", "bottom", "top", "right"):
                ax = plot.getAxis(axis_name)
                ax.setPen(AXIS_PEN)
                ax.setTextPen(TEXT_PEN)
            plot.showGrid(x=False, y=True, alpha=0.25)

        self._raw_plot = pg.PlotWidget(title="ECG (z-scored): blue = original, orange = reconstruction")
        _style_plot(self._raw_plot)
        self._raw_plot.setLabel("left", "amplitude (σ)")
        self._raw_plot.setLabel("bottom", "time (s)")
        self._raw_plot.setMinimumHeight(180)
        self._raw_plot.setMaximumHeight(360)
        self._raw_curve = self._raw_plot.plot(pen=pg.mkPen("#1f77b4", width=1.6))
        self._recon_curve = self._raw_plot.plot(pen=pg.mkPen("#ff7f0e", width=1.6))
        self._anomaly_regions: list = []
        layout.addWidget(self._raw_plot, stretch=3)

        self._res_plot = pg.PlotWidget(
            title="Anomaly score per beat-centred 2-s window  (p99 = 99th "
                  "percentile of the per-sample squared reconstruction error)"
        )
        _style_plot(self._res_plot)
        self._res_plot.setLabel("left", "p99 score")
        self._res_plot.setLabel("bottom", "window index (newest on the right)")
        self._res_plot.setToolTip(
            "<b>Anomaly score = p99</b> (99th percentile of the per-sample "
            "squared reconstruction error) of the trained autoencoder, "
            "computed for every beat-centred 2-second window. The dashed red "
            "line is the calibrated decision threshold τ = 0.3096 (95%-"
            "sensitivity operating point on the test split). When 2 of the "
            "last 3 windows exceed the threshold the system enters the "
            "ANOMALY state."
        )
        self._res_plot.setMinimumHeight(120)
        self._res_plot.setMaximumHeight(220)
        # Dark blue, thick — readable on white in a screenshot
        self._res_curve = self._res_plot.plot(pen=pg.mkPen("#1f3d6e", width=2.0))
        # Threshold: solid red, thick dashed line — high contrast against the curve
        self._thr_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#d62728", width=2.0, style=Qt.DashLine),
        )
        self._res_plot.addItem(self._thr_line)
        layout.addWidget(self._res_plot, stretch=2)

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
            # Cap as a safety net — if anomaly_end never arrives (e.g.
            # the buffer empties), we still won't fill the chart with red.
            while len(self._anomaly_regions) > MAX_ANOMALY_REGIONS:
                old = self._anomaly_regions.pop(0)
                try:
                    self._raw_plot.removeItem(old)
                except Exception:  # pragma: no cover - defensive
                    pass
        elif edge.event_type == "anomaly_end":
            # Falling edge: remove the most-recent (still-displayed) region
            # so the chart returns to its calm state once the episode ends.
            # Push/pop pairing keeps overlapping episodes correct.
            if self._anomaly_regions:
                old = self._anomaly_regions.pop()
                try:
                    self._raw_plot.removeItem(old)
                except Exception:  # pragma: no cover - defensive
                    pass
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
