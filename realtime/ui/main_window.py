"""Main application window with start/stop and one StreamPanel per patient."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QMainWindow, QPushButton, QStatusBar, QVBoxLayout, QWidget,
)

from realtime.config_rt import RealtimeConfig
from realtime.notifier import PipelineSignals
from realtime.ui.stream_panel import StreamPanel
from realtime.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        cfg: RealtimeConfig,
        signals: PipelineSignals,
        on_start: Callable[[RealtimeConfig], None],
        on_stop: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("ECG Anomaly Monitor")
        self._cfg = cfg
        self._signals = signals
        self._on_start = on_start
        self._on_stop = on_stop

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._panels: dict = {}
        for pid in cfg.records:
            panel = StreamPanel(pid, cfg.sampling_rate, cfg.stride_samples)
            layout.addWidget(panel)
            self._panels[pid] = panel

        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._do_start)
        self._stop_btn.clicked.connect(self._do_stop)
        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)

        self.setStatusBar(QStatusBar())
        self._wire_menu()
        self._wire_signals()

    def _wire_menu(self) -> None:
        act = QAction("Settings…", self)
        act.triggered.connect(self._open_settings)
        self.menuBar().addAction(act)

    def _wire_signals(self) -> None:
        s = self._signals
        s.new_window.connect(self._fan_out_window)
        s.detection.connect(self._fan_out_detection)
        s.anomaly_edge.connect(self._fan_out_edge)

    def _fan_out_window(self, result) -> None:
        panel = self._panels.get(result.patient_id)
        if panel:
            panel.on_window(result)

    def _fan_out_detection(self, d) -> None:
        panel = self._panels.get(d.patient_id)
        if panel:
            panel.on_detection(d)

    def _fan_out_edge(self, edge) -> None:
        panel = self._panels.get(edge.patient_id)
        if panel:
            panel.on_edge(edge)
        self.statusBar().showMessage(
            f"{edge.patient_id}: {edge.event_type} at {edge.ts_utc}", 5000
        )

    def _do_start(self) -> None:
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._on_start(self._cfg)

    def _do_stop(self) -> None:
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._on_stop()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec_() == dlg.Accepted:
            self._cfg = dlg.result_config()
            self.statusBar().showMessage(
                "Settings updated. Restart pipeline to apply.", 5000
            )
