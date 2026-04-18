import time

import numpy as np
import pytest
import torch
from torch import nn

from realtime.config_rt import RealtimeConfig
from realtime.event_store import EventStore
from realtime.pipeline import Pipeline


class BiasedModel(nn.Module):
    """Zero residual for the first few calls (warmup), then large residual."""
    def __init__(self, warmup_calls: int = 3):
        super().__init__()
        self._calls = 0
        self._warmup_calls = warmup_calls

    def forward(self, x):
        self._calls += 1
        if self._calls <= self._warmup_calls:
            return x
        return x + 5.0


def test_pipeline_fires_anomaly_in_headless_mode(tmp_path, monkeypatch):
    class FakeRec:
        p_signal = np.zeros((6 * 360, 1), dtype=float)
        fs = 360

    import realtime.stream_source as ss
    monkeypatch.setattr(ss, "_rdrecord", lambda _p: FakeRec())

    cfg = RealtimeConfig(
        residual_buffer_w=10, warmup_min=3,
        smoother_k=1, smoother_m=1,
        threshold_mode="percentile", percentile_q=0.5,
        records=("100",), inference_workers=1,
    )
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    store.start_writer_thread()

    pipeline = Pipeline(cfg, model=BiasedModel(), event_store=store,
                        signals=None, headless=True, record_map={"100": "fake"})
    pipeline.start()
    deadline = time.time() + 6.0
    while time.time() < deadline:
        if pipeline.anomaly_count >= 1:
            break
        time.sleep(0.05)
    pipeline.stop()
    store.close()

    assert pipeline.anomaly_count >= 1
