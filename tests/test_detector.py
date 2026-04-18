import numpy as np
import pytest

from realtime.config_rt import RealtimeConfig
from realtime.detector import Detector, DetectionResult


def make_cfg(**over):
    base = dict(
        threshold_mode="percentile",
        percentile_q=0.99,
        residual_buffer_w=1000,
        warmup_min=10,
        smoother_k=2,
        smoother_m=3,
    )
    base.update(over)
    return RealtimeConfig(**{**RealtimeConfig().__dict__, **base})


def test_detector_warmup_emits_no_anomaly():
    cfg = make_cfg()
    det = Detector("100", cfg)
    for _ in range(5):
        r = det.observe(residual=0.01)
        assert r.event == "none"
        assert r.state == "warmup"


def test_detector_fires_on_sustained_exceedance(rng):
    cfg = make_cfg()
    det = Detector("100", cfg)

    # 500 small residuals to fill buffer past warmup; enough that p99 of
    # the full buffer stays well below the spike value.
    for v in rng.normal(0.01, 0.001, size=500):
        det.observe(residual=float(v))

    # Spike: three windows above p99
    results = [det.observe(residual=10.0) for _ in range(3)]
    # 2-of-3 rule: on second True (i=1) buffer is [0, T, T] -> 2 Trues -> rising
    assert any(r.event == "rising" for r in results)
    assert results[-1].state == "anomaly"


def test_detector_reports_falling_edge():
    cfg = make_cfg()
    det = Detector("100", cfg)
    for v in np.full(500, 0.01):
        det.observe(residual=float(v))
    for _ in range(3):
        det.observe(residual=10.0)
    # Now feed 3 quiet windows -> should de-confirm
    quiet = [det.observe(residual=0.01) for _ in range(3)]
    assert any(r.event == "falling" for r in quiet)


def test_observe_is_keyword_only():
    cfg = make_cfg()
    det = Detector("100", cfg)
    with pytest.raises(TypeError):
        det.observe(0.1)  # type: ignore[misc]
