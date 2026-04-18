"""Per-stream anomaly detector: threshold + N-of-M + edge detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from realtime.config_rt import RealtimeConfig
from realtime.smoother import NofMSmoother
from realtime.threshold import make_threshold, Threshold

State = Literal["warmup", "normal", "anomaly"]
EdgeEvent = Literal["none", "rising", "falling"]


@dataclass(frozen=True)
class DetectionResult:
    patient_id: str
    residual: float
    threshold: float | None
    exceeded: bool
    state: State
    event: EdgeEvent


class Detector:
    def __init__(self, patient_id: str, cfg: RealtimeConfig) -> None:
        self._patient_id = patient_id
        self._cfg = cfg
        self._smoother = NofMSmoother(cfg.smoother_k, cfg.smoother_m)
        self._threshold: Threshold = self._build_threshold(cfg)

    def _build_threshold(self, cfg: RealtimeConfig) -> Threshold:
        kwargs = {}
        if cfg.threshold_mode == "percentile":
            kwargs = {"q": cfg.percentile_q}
        elif cfg.threshold_mode == "zscore":
            kwargs = {"k": cfg.zscore_k}
        elif cfg.threshold_mode == "fixed_online":
            kwargs = {
                "calibration_path": cfg.calibration_path,
                "nudge": cfg.fixed_online_nudge,
            }
        return make_threshold(
            cfg.threshold_mode,
            w=cfg.residual_buffer_w,
            warmup_min=cfg.warmup_min,
            **kwargs,
        )

    def observe(self, *, residual: float) -> DetectionResult:
        self._threshold.observe(residual)
        thr = self._threshold.current()
        if thr is None:
            return DetectionResult(
                patient_id=self._patient_id,
                residual=residual,
                threshold=None,
                exceeded=False,
                state="warmup",
                event="none",
            )
        exceeded = residual > thr
        event = self._smoother.push(exceeded)
        state: State = "anomaly" if self._smoother.confirmed else "normal"
        return DetectionResult(
            patient_id=self._patient_id,
            residual=residual,
            threshold=thr,
            exceeded=exceeded,
            state=state,
            event=event,
        )
