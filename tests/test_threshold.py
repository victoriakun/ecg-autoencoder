import json
from pathlib import Path

import numpy as np
import pytest

from realtime.threshold import (
    PercentileThreshold,
    ZScoreThreshold,
    FixedOnlineThreshold,
    make_threshold,
)


def test_percentile_warms_up_then_returns_quantile():
    t = PercentileThreshold(q=0.99, w=100, warmup_min=10)
    # Before warmup, returns None
    for _ in range(9):
        t.observe(0.1)
    assert t.current() is None

    t.observe(0.1)  # 10th sample
    assert t.current() is not None

    # Fill buffer with 100 samples drawn from a known distribution
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=100)
    t = PercentileThreshold(q=0.99, w=100, warmup_min=10)
    for v in values:
        t.observe(float(v))
    expected = float(np.quantile(values, 0.99))
    assert abs(t.current() - expected) < 1e-6


def test_zscore_threshold():
    t = ZScoreThreshold(k=3.0, w=50, warmup_min=10)
    for v in np.full(50, 1.0):
        t.observe(float(v))
    # std=0 -> threshold = mean + k*0 = 1.0
    assert abs(t.current() - 1.0) < 1e-6

    t = ZScoreThreshold(k=3.0, w=50, warmup_min=10)
    rng = np.random.default_rng(1)
    vals = rng.normal(0.5, 0.2, size=50)
    for v in vals:
        t.observe(float(v))
    expected = float(np.mean(vals) + 3.0 * np.std(vals))
    assert abs(t.current() - expected) < 1e-6


def test_fixed_online_starts_from_calibration(tmp_path: Path):
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"p99": 0.42}))
    t = FixedOnlineThreshold(calibration_path=calib, nudge=0.05, w=50, warmup_min=10)
    # Before warmup, returns the calibrated value
    assert abs(t.current() - 0.42) < 1e-6


def test_fixed_online_nudges_with_recent_median(tmp_path: Path):
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"p99": 1.0}))
    t = FixedOnlineThreshold(calibration_path=calib, nudge=0.05, w=20, warmup_min=10)
    for _ in range(20):
        t.observe(0.0)
    # Median is 0 -> threshold nudged DOWN by 5%
    assert abs(t.current() - 0.95) < 1e-6


def test_make_threshold_factory(tmp_path: Path):
    p = make_threshold("percentile", w=100, warmup_min=10, q=0.99)
    assert isinstance(p, PercentileThreshold)
    z = make_threshold("zscore", w=100, warmup_min=10, k=3.0)
    assert isinstance(z, ZScoreThreshold)
    calib = tmp_path / "c.json"
    calib.write_text('{"p99": 0.5}')
    f = make_threshold("fixed_online", w=100, warmup_min=10,
                       calibration_path=calib, nudge=0.05)
    assert isinstance(f, FixedOnlineThreshold)
    with pytest.raises(ValueError):
        make_threshold("bogus", w=100, warmup_min=10)
