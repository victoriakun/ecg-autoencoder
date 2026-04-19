import numpy as np
import pytest

from realtime.normalizer import WarmupNormalizer


def test_returns_none_during_warmup():
    n = WarmupNormalizer(warmup_samples=1000)
    out = n.observe(np.ones(720))
    assert out is None
    assert not n.is_ready


def test_freezes_stats_once_warmup_complete():
    n = WarmupNormalizer(warmup_samples=720)
    rng = np.random.default_rng(0)
    chunk = rng.standard_normal(720) * 2.0 + 5.0
    out = n.observe(chunk)
    assert out is not None
    assert n.is_ready
    mean, std = n.stats
    assert abs(mean - chunk.mean()) < 1e-6
    assert abs(std - (chunk.std() + 1e-8)) < 1e-6


def test_subsequent_windows_use_frozen_stats():
    n = WarmupNormalizer(warmup_samples=720)
    n.observe(np.zeros(720) + 10.0)
    out = n.observe(np.full(720, 12.0))
    expected = (12.0 - 10.0) / (0.0 + 1e-8)
    assert np.allclose(out, expected)


def test_buffer_truncated_to_target():
    n = WarmupNormalizer(warmup_samples=500)
    n.observe(np.arange(720, dtype=float))
    assert n.is_ready
    mean, _ = n.stats
    # Mean should equal mean of first 500 samples (0..499) = 249.5
    assert abs(mean - 249.5) < 1e-6


def test_invalid_warmup():
    with pytest.raises(ValueError):
        WarmupNormalizer(warmup_samples=0)
