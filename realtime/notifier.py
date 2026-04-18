"""Qt signals fanned out to the UI layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt5.QtCore import QObject, pyqtSignal

from realtime.detector import DetectionResult
from realtime.inference import InferenceResult


@dataclass(frozen=True)
class AnomalyEdge:
    patient_id: str
    event_type: str
    ts_utc: str
    residual: float
    threshold: float


class PipelineSignals(QObject):
    new_window = pyqtSignal(object)
    detection = pyqtSignal(object)
    anomaly_edge = pyqtSignal(object)
    health = pyqtSignal(dict)
    stopped = pyqtSignal()

    def emit_window(self, r: InferenceResult) -> None:
        self.new_window.emit(r)

    def emit_detection(self, d: DetectionResult) -> None:
        self.detection.emit(d)

    def emit_edge(self, e: AnomalyEdge) -> None:
        self.anomaly_edge.emit(e)

    def emit_health(self, metrics: dict[str, Any]) -> None:
        self.health.emit(metrics)

    def emit_stopped(self) -> None:
        self.stopped.emit()
