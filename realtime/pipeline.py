"""Supervises producers, inference pool, detectors, and the event store."""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from torch import nn

from realtime.config_rt import RealtimeConfig
from realtime.detector import Detector, DetectionResult
from realtime.event_store import AnomalyEvent, EventStore, ModelVersionRecord
from realtime.inference import InferenceResult, InferenceWorker
from realtime.normalizer import WarmupNormalizer
from realtime.stream_source import StreamSource, StreamWindow

log = logging.getLogger(__name__)


class _SystemClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


class Pipeline:
    def __init__(
        self,
        cfg: RealtimeConfig,
        model: nn.Module,
        event_store: EventStore,
        signals=None,
        headless: bool = False,
        record_map: Optional[dict] = None,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._store = event_store
        self._signals = signals
        self._headless = headless
        self._record_map = record_map or {r: f"data/mitbih/{r}" for r in cfg.records}

        self._stop = threading.Event()
        self._windows_q: queue.Queue = queue.Queue(maxsize=cfg.queue_maxsize)
        self._results_q: queue.Queue = queue.Queue(maxsize=cfg.queue_maxsize)

        self._detectors: dict = {r: Detector(r, cfg) for r in cfg.records}
        if cfg.normalizer_warmup_samples > 0:
            self._normalizers: dict = {
                r: WarmupNormalizer(cfg.normalizer_warmup_samples)
                for r in cfg.records
            }
        else:
            self._normalizers = None
        self._threads: list = []
        self._model_version = self._compute_model_version()
        self.anomaly_count = 0

    def _compute_model_version(self) -> str:
        try:
            path = Path(self._cfg.model_path)
            if path.exists():
                sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            else:
                sha = "unknown"
        except Exception:
            sha = "unknown"
        try:
            git = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git = "unknown"
        return f"{Path(self._cfg.model_path).name}@sha={sha}@git={git}"

    def start(self) -> None:
        self._store.log_model_version(ModelVersionRecord(
            loaded_at_utc=_now_iso(),
            model_path=self._cfg.model_path,
            model_sha256=self._model_version.split("sha=")[1].split("@")[0],
            git_commit=self._model_version.split("git=")[1],
            config_snapshot=json.dumps(asdict(self._cfg), default=str),
        ))

        clock = _SystemClock()
        for patient_id in self._cfg.records:
            record_path = self._record_map.get(patient_id, patient_id)
            src = StreamSource(
                patient_id=patient_id,
                record_path=record_path,
                window_samples=self._cfg.window_samples,
                stride_samples=self._cfg.stride_samples,
                clock=clock,
                out_queue=self._windows_q,
                stop_event=self._stop,
                windowing_mode=self._cfg.windowing_mode,
            )
            self._spawn(f"src-{patient_id}", src.run)

        for i in range(self._cfg.inference_workers):
            w = InferenceWorker(
                self._model, self._windows_q, self._results_q, self._stop,
                sampling_rate=self._cfg.sampling_rate,
                normalizers=self._normalizers,
            )
            self._spawn(f"inf-{i}", w.run)

        self._spawn("detector", self._dispatch_loop)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._results_q.get(timeout=0.5)
            except queue.Empty:
                continue
            det = self._detectors.get(result.patient_id)
            if det is None:
                log.warning("no detector for patient %s", result.patient_id)
                continue
            detection = det.observe(residual=result.residual)
            if self._signals is not None:
                self._signals.emit_window(result)
                self._signals.emit_detection(detection)
            if detection.event in ("rising", "falling"):
                self._handle_edge(result, detection)

    def _handle_edge(self, result: InferenceResult, d: DetectionResult) -> None:
        event_type = "anomaly_start" if d.event == "rising" else "anomaly_end"
        offset_seconds = (
            float(result.record_offset_samples) / float(self._cfg.sampling_rate)
            if hasattr(result, "record_offset_samples") else None
        )
        # On rising edges we persist the exact 720-sample window that fired
        # the alert plus the model's reconstruction of that window AND the
        # index of the per-sample peak error, so the cardiologist can review
        # the alert later in the UI with the original-vs-reconstruction
        # overlay required by the system-design chapter.
        waveform = None
        reconstruction = None
        peak_idx = None
        if event_type == "anomaly_start" and result.raw is not None:
            try:
                import numpy as np
                waveform = np.asarray(result.raw, dtype=np.float32)
                reconstruction = np.asarray(result.recon, dtype=np.float32)
                err = (waveform - reconstruction) ** 2
                peak_idx = int(err.argmax())
            except Exception as e:  # pragma: no cover - defensive
                log.warning("could not compute peak index for anomaly: %s", e)
                waveform, reconstruction, peak_idx = None, None, None
        ev = AnomalyEvent(
            patient_id=d.patient_id,
            event_type=event_type,
            ts_utc=result.ts_utc,
            residual=d.residual,
            threshold=d.threshold or 0.0,
            threshold_mode=self._cfg.threshold_mode,
            model_version=self._model_version,
            record_offset_seconds=offset_seconds,
            waveform=waveform,
            reconstruction=reconstruction,
            peak_sample_index=peak_idx,
        )
        self._store.queue_event(ev)
        if event_type == "anomaly_start":
            self.anomaly_count += 1
        if self._signals is not None:
            from realtime.notifier import AnomalyEdge
            self._signals.emit_edge(AnomalyEdge(
                patient_id=d.patient_id,
                event_type=event_type,
                ts_utc=result.ts_utc,
                residual=d.residual,
                threshold=d.threshold or 0.0,
            ))

    def _spawn(self, name: str, target) -> None:
        def _wrapped():
            try:
                target()
            except Exception as e:
                log.error("thread %s crashed: %s", name, e)

        t = threading.Thread(target=_wrapped, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        if self._signals is not None:
            self._signals.emit_stopped()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
