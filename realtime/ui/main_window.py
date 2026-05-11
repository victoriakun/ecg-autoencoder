"""Main application window with start/stop and one StreamPanel per patient."""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QHBoxLayout, QMainWindow, QPushButton, QScrollArea,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from realtime.config_rt import RealtimeConfig
from realtime.event_store import EventStore
from realtime.notifier import PipelineSignals
from realtime.ui.review_dialog import ReviewDialog
from realtime.ui.stream_panel import StreamPanel
from realtime.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        cfg: RealtimeConfig,
        signals: PipelineSignals,
        on_start: Callable[[RealtimeConfig], None],
        on_stop: Callable[[], None],
        event_store: Optional[EventStore] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("ECG Anomaly Monitor")
        self._cfg = cfg
        self._signals = signals
        self._on_start = on_start
        self._on_stop = on_stop
        self._event_store = event_store

        # Top toolbar: Start / Stop / Review — always visible regardless of
        # how many StreamPanels are stacked below.
        toolbar = QToolBar("Pipeline controls")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        self._start_btn = QPushButton("▶  Start")
        self._stop_btn = QPushButton("■  Stop")
        self._review_btn = QPushButton("📋  Review flagged anomalies…")
        self._stop_btn.setEnabled(False)
        # Disable the review button when no event store is wired in
        # (e.g. unit-test launches with signals only).
        self._review_btn.setEnabled(self._event_store is not None)
        self._start_btn.clicked.connect(self._do_start)
        self._stop_btn.clicked.connect(self._do_stop)
        self._review_btn.clicked.connect(self._open_review_dialog)
        for b in (self._start_btn, self._stop_btn):
            b.setMinimumWidth(120)
        self._review_btn.setMinimumWidth(220)
        toolbar.addWidget(self._start_btn)
        toolbar.addWidget(self._stop_btn)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Expanding, spacer.sizePolicy().Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self._review_btn)

        # Central area: a scrollable column of StreamPanels — one per record.
        # Wrapping in a QScrollArea means the toolbar above stays put even
        # when many panels are stacked.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panels_container = QWidget()
        panels_layout = QVBoxLayout(panels_container)
        panels_layout.setContentsMargins(8, 8, 8, 8)

        self._panels: dict = {}
        for pid in cfg.records:
            panel = StreamPanel(pid, cfg.sampling_rate, cfg.stride_samples)
            panels_layout.addWidget(panel)
            self._panels[pid] = panel
        panels_layout.addStretch()

        scroll.setWidget(panels_container)
        self.setCentralWidget(scroll)

        self.setStatusBar(QStatusBar())
        # Surface the loaded record list in the status bar so the user
        # knows immediately why N panels appeared.
        self.statusBar().showMessage(
            f"Loaded {len(cfg.records)} record(s): {', '.join(cfg.records)}  "
            f"— pass --records to choose, or --config <file>.json to override."
        )
        self._wire_menu()
        self._wire_signals()

    def _wire_menu(self) -> None:
        self.menuBar().setNativeMenuBar(False)
        menu = self.menuBar().addMenu("&File")
        act = QAction("Settings…", self)
        act.setMenuRole(QAction.NoRole)
        act.triggered.connect(self._open_settings)
        menu.addAction(act)

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

    def _open_review_dialog(self) -> None:
        if self._event_store is None:
            self.statusBar().showMessage(
                "No event store available — start the pipeline first.", 4000
            )
            return
        dlg = ReviewDialog(self._event_store, self)
        dlg.exec_()
