"""Runtime configuration for the real-time ECG pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ThresholdMode = Literal["percentile", "zscore", "fixed_online"]
VALID_MODES = ("percentile", "zscore", "fixed_online")


@dataclass(frozen=True)
class RealtimeConfig:
    sampling_rate: int = 360
    window_samples: int = 720          # 2 s @ 360 Hz (matches training)
    stride_samples: int = 180          # 0.5 s stride -> 75% overlap
    threshold_mode: ThresholdMode = "percentile"
    percentile_q: float = 0.99
    zscore_k: float = 3.0
    fixed_online_nudge: float = 0.05   # +/-5 % adjustment around the calibrated p99
    smoother_k: int = 2
    smoother_m: int = 3
    residual_buffer_w: int = 600       # 600 samples x 0.5 s = 5 min
    warmup_min: int = 60
    inference_workers: int = 2
    queue_maxsize: int = 8
    model_path: str = "models/ecg_autoencoder.pt"
    calibration_path: str = "models/ecg_autoencoder.calibration.json"
    records: tuple[str, ...] = ("100", "208", "222")

    def __post_init__(self) -> None:
        if self.threshold_mode not in VALID_MODES:
            raise ValueError(
                f"threshold_mode must be one of {VALID_MODES}, got {self.threshold_mode!r}"
            )
        if self.smoother_k > self.smoother_m:
            raise ValueError("smoother_k cannot exceed smoother_m")
        if self.stride_samples <= 0 or self.stride_samples > self.window_samples:
            raise ValueError("stride_samples must be in (0, window_samples]")


def save_config(cfg: RealtimeConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))


def load_config(path: Path) -> RealtimeConfig:
    data = json.loads(Path(path).read_text())
    # tuple conversion for JSON compatibility
    if isinstance(data.get("records"), list):
        data["records"] = tuple(data["records"])
    return RealtimeConfig(**data)
