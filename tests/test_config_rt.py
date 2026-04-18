from pathlib import Path

import pytest

from realtime.config_rt import RealtimeConfig, load_config, save_config


def test_defaults_match_spec():
    c = RealtimeConfig()
    assert c.window_samples == 720
    assert c.stride_samples == 180
    assert c.sampling_rate == 360
    assert c.threshold_mode == "percentile"
    assert c.percentile_q == 0.99
    assert c.zscore_k == 3.0
    assert c.smoother_k == 2
    assert c.smoother_m == 3
    assert c.residual_buffer_w == 600
    assert c.warmup_min == 60
    assert c.inference_workers == 2
    assert c.queue_maxsize == 8


def test_roundtrip(tmp_path: Path):
    c = RealtimeConfig(threshold_mode="zscore", zscore_k=2.5)
    path = tmp_path / "cfg.json"
    save_config(c, path)
    loaded = load_config(path)
    assert loaded == c


def test_invalid_threshold_mode_rejected():
    with pytest.raises(ValueError):
        RealtimeConfig(threshold_mode="nope")


def test_smoother_k_cannot_exceed_m():
    with pytest.raises(ValueError):
        RealtimeConfig(smoother_k=4, smoother_m=3)
