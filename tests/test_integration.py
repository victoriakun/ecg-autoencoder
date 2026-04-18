from pathlib import Path

import pytest
import torch

from realtime.config_rt import RealtimeConfig
from realtime.event_store import EventStore
from realtime.inference import load_model
from realtime.pipeline import Pipeline
from models import ConvAutoencoder


REC_PATH = Path("data/mitbih/208")


@pytest.mark.skipif(
    not REC_PATH.with_suffix(".dat").exists(),
    reason="MIT-BIH record 208 not present; run `python dataset.py --download`",
)
def test_record_208_produces_anomalies(tmp_path):
    cfg = RealtimeConfig(
        residual_buffer_w=200, warmup_min=30,
        smoother_k=2, smoother_m=3,
        threshold_mode="percentile", percentile_q=0.98,
        records=("208",), inference_workers=1,
        stride_samples=180,
    )
    store = EventStore(tmp_path / "events.db")
    store.start_writer_thread()
    model = load_model(Path(cfg.model_path), ConvAutoencoder)
    pipeline = Pipeline(cfg, model=model, event_store=store,
                        signals=None, headless=True)
    pipeline.start()

    import time
    time.sleep(30.0)
    pipeline.stop()
    store.close()

    assert pipeline.anomaly_count >= 1
