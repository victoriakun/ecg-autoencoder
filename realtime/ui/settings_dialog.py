"""Dialog for live-editing RealtimeConfig."""
from __future__ import annotations

from dataclasses import replace

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QSpinBox,
)

from realtime.config_rt import RealtimeConfig, VALID_MODES


class SettingsDialog(QDialog):
    def __init__(self, cfg: RealtimeConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Real-time settings")
        self._cfg = cfg
        form = QFormLayout(self)

        self._mode = QComboBox()
        self._mode.addItems(list(VALID_MODES))
        self._mode.setCurrentText(cfg.threshold_mode)

        self._q = QDoubleSpinBox()
        self._q.setRange(0.5, 0.999); self._q.setDecimals(3)
        self._q.setValue(cfg.percentile_q)

        self._k = QDoubleSpinBox()
        self._k.setRange(0.5, 10.0); self._k.setValue(cfg.zscore_k)

        self._sk = QSpinBox()
        self._sk.setRange(1, 10); self._sk.setValue(cfg.smoother_k)

        self._sm = QSpinBox()
        self._sm.setRange(1, 20); self._sm.setValue(cfg.smoother_m)

        self._stride = QSpinBox()
        self._stride.setRange(10, cfg.window_samples)
        self._stride.setSingleStep(10); self._stride.setValue(cfg.stride_samples)

        form.addRow("Threshold mode", self._mode)
        form.addRow("Percentile q", self._q)
        form.addRow("Z-score k", self._k)
        form.addRow("Smoother K", self._sk)
        form.addRow("Smoother M", self._sm)
        form.addRow("Stride (samples)", self._stride)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_config(self) -> RealtimeConfig:
        return replace(
            self._cfg,
            threshold_mode=self._mode.currentText(),
            percentile_q=self._q.value(),
            zscore_k=self._k.value(),
            smoother_k=self._sk.value(),
            smoother_m=self._sm.value(),
            stride_samples=self._stride.value(),
        )
