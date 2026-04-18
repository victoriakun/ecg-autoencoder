# Real-time ECG Anomaly Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing batch-trained ConvAutoencoder ECG project with a live pipeline that replays MIT-BIH records at 360 Hz, detects anomalies in sub-second time using dynamic thresholding, logs events to SQLite, and visualises 2–3 concurrent streams in a PyQt desktop UI.

**Architecture:** Threaded producer/consumer with Qt signals. Stream sources produce overlapping windows → shared inference thread pool runs the autoencoder → per-stream `Detector` applies threshold + N-of-M smoother → events go to SQLite writer thread and Qt UI. Config-driven, testable, headless mode for CI.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, pyqtgraph, PyQt5, SQLite (stdlib), `cryptography` (log encryption), pytest, wfdb, Docker.

**Spec:** `docs/superpowers/specs/2026-04-18-realtime-ecg-pipeline-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `realtime/__init__.py` | Package marker |
| `realtime/config_rt.py` | `RealtimeConfig` dataclass, loader/saver |
| `realtime/smoother.py` | `NofMSmoother` — pure logic, no deps |
| `realtime/threshold.py` | `PercentileThreshold`, `ZScoreThreshold`, `FixedOnlineThreshold` |
| `realtime/detector.py` | `Detector` — integrates threshold + smoother + edge detection |
| `realtime/stream_source.py` | `StreamSource` — replays MIT-BIH at 360 Hz, `FastClock` injection for tests |
| `realtime/inference.py` | `InferenceWorker` — loads model, computes residuals |
| `realtime/event_store.py` | SQLite schema + writer thread, model-version logging |
| `realtime/log_crypto.py` | Fernet key management + encrypt-on-rotate hook |
| `realtime/notifier.py` | Qt signals that fan out to UI slots |
| `realtime/pipeline.py` | Thread supervisor + queue wiring |
| `realtime/ui/__init__.py` | Package marker |
| `realtime/ui/stream_panel.py` | `StreamPanel` widget |
| `realtime/ui/main_window.py` | `MainWindow` composition |
| `realtime/ui/settings_dialog.py` | Settings dialog bound to `RealtimeConfig` |
| `realtime_app.py` | Entry point (GUI and `--headless` modes) |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Pytest fixtures (FastClock, fake model, tmp SQLite) |
| `tests/test_smoother.py` | |
| `tests/test_threshold.py` | |
| `tests/test_detector.py` | |
| `tests/test_stream_source.py` | |
| `tests/test_inference.py` | |
| `tests/test_event_store.py` | |
| `tests/test_log_crypto.py` | |
| `tests/test_integration.py` | Full-pipeline headless test on record 208 |
| `tests/test_latency.py` | Gated perf test |
| `Dockerfile` | |
| `docker-compose.yml` | |
| `docs/running-in-docker.md` | |
| `docs/manual-test.md` | UI smoke-test checklist |
| `logs/.gitkeep` | |

### Modified files

| Path | Change |
|---|---|
| `requirements.txt` | Add `pyqtgraph`, `PyQt5`, `cryptography`, `pytest`, `pytest-qt` (optional), `wfdb` (already there), `plyer` (optional desktop notifications) |
| `config.py` | Add reference pointer to `RealtimeConfig` only — keep batch constants untouched |
| `README.md` | Add "Real-time pipeline" section with usage snippet |
| `.gitignore` | Add `logs/events.db`, `logs/*.log*`, `logs/.key` |

---

## Task 1: Project scaffolding and pytest bootstrap

**Files:**
- Create: `realtime/__init__.py`, `realtime/ui/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `pytest.ini`, `logs/.gitkeep`
- Modify: `requirements.txt`, `.gitignore`

- [ ] **Step 1: Add runtime dependencies**

Append to `requirements.txt` (create if missing, preserving existing lines):

```
pyqtgraph>=0.13
PyQt5>=5.15
cryptography>=42
pytest>=8
plyer>=2.1
```

- [ ] **Step 2: Update .gitignore**

Append to `.gitignore`:

```
logs/events.db
logs/*.log
logs/*.log.*
logs/.key
realtime/__pycache__/
tests/__pycache__/
```

- [ ] **Step 3: Create package markers and logs dir**

Create empty files `realtime/__init__.py`, `realtime/ui/__init__.py`, `tests/__init__.py`, `logs/.gitkeep`.

- [ ] **Step 4: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q --tb=short
markers =
    latency: performance regression test, skipped unless TORCH_CPU_BENCHMARK=1
```

- [ ] **Step 5: Create tests/conftest.py with shared fixtures**

```python
"""Shared pytest fixtures for real-time pipeline tests."""
from __future__ import annotations

import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


class FastClock:
    """Deterministic clock for streaming tests. Advances only when tick() is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._t += seconds

    def tick(self, seconds: float) -> None:
        self.sleep(seconds)


@pytest.fixture
def fast_clock() -> FastClock:
    return FastClock()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "events.db"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
```

- [ ] **Step 6: Install deps and verify pytest runs**

Run: `pip install -r requirements.txt && pytest --collect-only`
Expected: exits 0, says "no tests ran".

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore realtime/ tests/ logs/.gitkeep pytest.ini
git commit -m "feat(realtime): scaffold package, pytest, shared fixtures"
```

---

## Task 2: RealtimeConfig dataclass

**Files:**
- Create: `realtime/config_rt.py`, `tests/test_config_rt.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config_rt.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config_rt.py -v`
Expected: import error — `realtime.config_rt` doesn't exist yet.

- [ ] **Step 3: Implement RealtimeConfig**

`realtime/config_rt.py`:

```python
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
    stride_samples: int = 180          # 0.5 s stride → 75% overlap
    threshold_mode: ThresholdMode = "percentile"
    percentile_q: float = 0.99
    zscore_k: float = 3.0
    fixed_online_nudge: float = 0.05   # ±5 % adjustment around the calibrated p99
    smoother_k: int = 2
    smoother_m: int = 3
    residual_buffer_w: int = 600       # 600 samples × 0.5 s = 5 min
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_rt.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/config_rt.py tests/test_config_rt.py
git commit -m "feat(realtime): add RealtimeConfig dataclass with json roundtrip"
```

---

## Task 3: N-of-M smoother

**Files:**
- Create: `realtime/smoother.py`, `tests/test_smoother.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_smoother.py`:

```python
from realtime.smoother import NofMSmoother


def test_initial_state_is_false():
    s = NofMSmoother(k=2, m=3)
    assert s.confirmed is False


def test_single_exceedance_not_confirmed():
    s = NofMSmoother(k=2, m=3)
    s.push(True)
    assert s.confirmed is False


def test_two_of_three_confirms():
    s = NofMSmoother(k=2, m=3)
    s.push(True)
    s.push(False)
    s.push(True)
    assert s.confirmed is True


def test_falling_edge_after_enough_negatives():
    s = NofMSmoother(k=2, m=3)
    for v in [True, True, True]:
        s.push(v)
    assert s.confirmed is True
    s.push(False)
    s.push(False)
    # Now window is [True, False, False] → confirmed = False
    assert s.confirmed is False


def test_rising_edge_event_reported_once():
    s = NofMSmoother(k=2, m=3)
    assert s.push(False) == "none"
    assert s.push(True) == "none"
    assert s.push(True) == "rising"       # first time confirmed
    assert s.push(True) == "none"         # still confirmed, no new edge
    assert s.push(False) == "none"
    assert s.push(False) == "falling"     # de-confirmed


def test_k_equals_m():
    s = NofMSmoother(k=3, m=3)
    s.push(True); s.push(True); s.push(False)
    assert s.confirmed is False
    s.push(True); s.push(True); s.push(True)
    assert s.confirmed is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_smoother.py -v`
Expected: import error.

- [ ] **Step 3: Implement the smoother**

`realtime/smoother.py`:

```python
"""N-of-M smoother for anomaly confirmation."""
from __future__ import annotations

from collections import deque
from typing import Literal

EdgeEvent = Literal["none", "rising", "falling"]


class NofMSmoother:
    """Confirms an alarm when K of the last M decisions are True.

    Reports edge transitions via push(): "rising" when first confirmed,
    "falling" when de-confirmed, "none" otherwise.
    """

    def __init__(self, k: int, m: int) -> None:
        if not (1 <= k <= m):
            raise ValueError("require 1 <= k <= m")
        self._k = k
        self._buf: deque[bool] = deque(maxlen=m)
        self._confirmed = False

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def push(self, exceeded: bool) -> EdgeEvent:
        self._buf.append(bool(exceeded))
        new = sum(self._buf) >= self._k
        if new and not self._confirmed:
            self._confirmed = True
            return "rising"
        if self._confirmed and not new:
            self._confirmed = False
            return "falling"
        return "none"
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_smoother.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/smoother.py tests/test_smoother.py
git commit -m "feat(realtime): add N-of-M smoother with edge reporting"
```

---

## Task 4: Threshold strategies

**Files:**
- Create: `realtime/threshold.py`, `tests/test_threshold.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_threshold.py`:

```python
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
    # std=0 → threshold = mean + k*0 = 1.0
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
    # Median is 0 → threshold nudged DOWN by 5%
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
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_threshold.py -v`
Expected: import error.

- [ ] **Step 3: Implement threshold classes**

`realtime/threshold.py`:

```python
"""Dynamic threshold strategies for anomaly detection."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np


class Threshold(Protocol):
    def observe(self, residual: float) -> None: ...
    def current(self) -> float | None: ...


class _RollingBase:
    def __init__(self, w: int, warmup_min: int) -> None:
        if warmup_min > w:
            raise ValueError("warmup_min must be <= w")
        self._w = w
        self._warmup_min = warmup_min
        self._buf: deque[float] = deque(maxlen=w)

    def observe(self, residual: float) -> None:
        self._buf.append(float(residual))

    def _ready(self) -> bool:
        return len(self._buf) >= self._warmup_min


class PercentileThreshold(_RollingBase):
    def __init__(self, q: float, w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        if not 0 < q < 1:
            raise ValueError("q must be in (0, 1)")
        self._q = q

    def current(self) -> float | None:
        if not self._ready():
            return None
        return float(np.quantile(self._buf, self._q))


class ZScoreThreshold(_RollingBase):
    def __init__(self, k: float, w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        self._k = k

    def current(self) -> float | None:
        if not self._ready():
            return None
        arr = np.asarray(self._buf, dtype=float)
        return float(arr.mean() + self._k * arr.std())


class FixedOnlineThreshold(_RollingBase):
    """Start from calibrated p99, nudge by ±nudge based on recent median."""

    def __init__(self, calibration_path: Path, nudge: float,
                 w: int, warmup_min: int) -> None:
        super().__init__(w, warmup_min)
        payload = json.loads(Path(calibration_path).read_text())
        self._base = float(payload["p99"])
        self._nudge = nudge

    def current(self) -> float | None:
        if not self._ready():
            return self._base
        median = float(np.median(self._buf))
        ratio = median / self._base if self._base else 0.0
        # If recent median is low relative to base, reduce threshold
        if ratio < 1.0:
            return self._base * (1.0 - self._nudge)
        if ratio > 1.0:
            return self._base * (1.0 + self._nudge)
        return self._base


def make_threshold(mode: str, *, w: int, warmup_min: int, **kwargs) -> Threshold:
    if mode == "percentile":
        return PercentileThreshold(q=kwargs["q"], w=w, warmup_min=warmup_min)
    if mode == "zscore":
        return ZScoreThreshold(k=kwargs["k"], w=w, warmup_min=warmup_min)
    if mode == "fixed_online":
        return FixedOnlineThreshold(
            calibration_path=kwargs["calibration_path"],
            nudge=kwargs["nudge"],
            w=w, warmup_min=warmup_min,
        )
    raise ValueError(f"unknown threshold mode: {mode}")
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_threshold.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/threshold.py tests/test_threshold.py
git commit -m "feat(realtime): add three threshold strategies with factory"
```

---

## Task 5: Detector — edge events + smoother integration

**Files:**
- Create: `realtime/detector.py`, `tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_detector.py`:

```python
import numpy as np
import pytest

from realtime.config_rt import RealtimeConfig
from realtime.detector import Detector, DetectionResult


def make_cfg(**over):
    base = dict(
        threshold_mode="percentile",
        percentile_q=0.99,
        residual_buffer_w=100,
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

    # 50 small residuals to fill buffer past warmup
    for v in rng.normal(0.01, 0.001, size=50):
        det.observe(float(v))

    # Spike: three windows above p99
    results = [det.observe(10.0) for _ in range(3)]
    # 2-of-3 rule: on second True (i=1) buffer is [0, T, T] → 2 Trues → rising
    assert any(r.event == "rising" for r in results)
    assert results[-1].state == "anomaly"


def test_detector_reports_falling_edge():
    cfg = make_cfg()
    det = Detector("100", cfg)
    for v in np.full(50, 0.01):
        det.observe(float(v))
    for _ in range(3):
        det.observe(10.0)
    # Now feed 3 quiet windows → should de-confirm
    quiet = [det.observe(0.01) for _ in range(3)]
    assert any(r.event == "falling" for r in quiet)


def test_unknown_stream_id_rejected():
    cfg = make_cfg()
    det = Detector("100", cfg)
    with pytest.raises(ValueError):
        det.observe(0.1)  # positional call missing residual? (keyword-only)
```

Note: the last test is a signature/API contract check. If `observe(0.1)` is not rejected, adjust the test to whatever enforcement you add (e.g., require kwargs). The purpose is to lock down the API so later tasks don't drift.

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_detector.py -v`
Expected: import error.

- [ ] **Step 3: Implement Detector**

`realtime/detector.py`:

```python
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
```

Adjust the failing test's last case to pass a positional arg and assert the `TypeError` raised by Python's keyword-only enforcement:

```python
def test_observe_is_keyword_only():
    cfg = make_cfg()
    det = Detector("100", cfg)
    with pytest.raises(TypeError):
        det.observe(0.1)  # type: ignore[misc]
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_detector.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/detector.py tests/test_detector.py
git commit -m "feat(realtime): add Detector integrating threshold, smoother, edges"
```

---

## Task 6: StreamSource — replay MIT-BIH records at wall-clock pace

**Files:**
- Create: `realtime/stream_source.py`, `tests/test_stream_source.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_stream_source.py`:

```python
import queue
import threading
import time

import numpy as np
import pytest

from realtime.stream_source import StreamSource, StreamWindow


@pytest.fixture
def fake_record(monkeypatch):
    """Patch wfdb.rdrecord so no actual file is read."""
    class FakeRec:
        p_signal = np.arange(5 * 360, dtype=float).reshape(-1, 1)
        fs = 360

    def _rdrecord(path):
        return FakeRec()

    import realtime.stream_source as ss
    monkeypatch.setattr(ss, "_rdrecord", _rdrecord)
    return FakeRec


def test_emits_windows_at_expected_count(fake_record, fast_clock):
    q: queue.Queue[StreamWindow] = queue.Queue()
    stop = threading.Event()
    src = StreamSource(
        patient_id="100",
        record_path="irrelevant",
        window_samples=720,
        stride_samples=180,
        clock=fast_clock,
        out_queue=q,
        stop_event=stop,
    )
    # Signal length = 5s * 360 = 1800 samples. Windows of 720 with stride 180 →
    # (1800 - 720) / 180 + 1 = 7 windows.
    t = threading.Thread(target=src.run)
    t.start()

    # Advance fake clock enough for all windows (7 strides × 0.5 s)
    for _ in range(10):
        fast_clock.tick(0.5)
        time.sleep(0.001)  # let worker thread run

    stop.set()
    t.join(timeout=2.0)

    collected = []
    while not q.empty():
        collected.append(q.get_nowait())
    assert len(collected) == 7
    assert collected[0].patient_id == "100"
    assert collected[0].samples.shape == (720,)


def test_drops_short_windows(fake_record, fast_clock):
    """Signal length 700 (< 720) should emit zero windows with WARNING."""
    class ShortRec:
        p_signal = np.zeros((700, 1), dtype=float)
        fs = 360

    import realtime.stream_source as ss
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def _rdr(_):
        return ShortRec()
    ss._rdrecord = _rdr

    src = StreamSource("100", "x", 720, 180, fast_clock, q, stop)
    src.run()  # returns immediately
    assert q.empty()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_stream_source.py -v`
Expected: import error.

- [ ] **Step 3: Implement StreamSource**

`realtime/stream_source.py`:

```python
"""Replays a saved ECG record into a queue at wall-clock pace."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# Indirection so tests can monkeypatch
def _rdrecord(path: str):
    import wfdb
    return wfdb.rdrecord(path)


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class StreamWindow:
    patient_id: str
    ts_utc: str
    samples: np.ndarray  # shape (window_samples,)


class StreamSource:
    def __init__(
        self,
        patient_id: str,
        record_path: str | Path,
        window_samples: int,
        stride_samples: int,
        clock: Clock,
        out_queue: "queue.Queue[StreamWindow]",
        stop_event: threading.Event,
    ) -> None:
        self._patient_id = patient_id
        self._record_path = str(record_path)
        self._window = window_samples
        self._stride = stride_samples
        self._clock = clock
        self._out = out_queue
        self._stop = stop_event

    def run(self) -> None:
        try:
            rec = _rdrecord(self._record_path)
        except Exception as e:
            log.error("failed to load record %s: %s", self._record_path, e)
            return

        signal = np.asarray(rec.p_signal[:, 0], dtype=float)
        fs = getattr(rec, "fs", 360)
        if signal.size < self._window:
            log.warning(
                "record %s too short (%d samples < %d); skipping",
                self._record_path, signal.size, self._window,
            )
            return

        stride_seconds = self._stride / float(fs)
        start = 0
        while not self._stop.is_set():
            end = start + self._window
            if end > signal.size:
                break
            window = signal[start:end].copy()
            if np.isnan(window).any():
                log.warning("NaN in window at offset %d (%s); skipping",
                            start, self._patient_id)
            else:
                try:
                    self._out.put(
                        StreamWindow(
                            patient_id=self._patient_id,
                            ts_utc=_now_iso(),
                            samples=window,
                        ),
                        timeout=1.0,
                    )
                except queue.Full:
                    log.warning("queue full for %s; dropping window",
                                self._patient_id)
            start += self._stride
            self._clock.sleep(stride_seconds)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_stream_source.py -v`
Expected: 2 passed. If the first test is flaky due to thread scheduling, add a small `time.sleep` loop after each `tick()`.

- [ ] **Step 5: Commit**

```bash
git add realtime/stream_source.py tests/test_stream_source.py
git commit -m "feat(realtime): StreamSource replays MIT-BIH at stride pace"
```

---

## Task 7: Inference worker

**Files:**
- Create: `realtime/inference.py`, `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_inference.py`:

```python
import queue
import threading

import numpy as np
import torch

from realtime.inference import InferenceWorker, InferenceResult
from realtime.stream_source import StreamWindow


class IdentityModel(torch.nn.Module):
    """Returns input unchanged → residual is always zero."""
    def forward(self, x):
        return x


class BiasedModel(torch.nn.Module):
    """Returns input + 1 → residual = 1 for constant input of 0."""
    def forward(self, x):
        return x + 1.0


def test_identity_model_yields_zero_residual():
    in_q: queue.Queue[StreamWindow] = queue.Queue()
    out_q: queue.Queue[InferenceResult] = queue.Queue()
    stop = threading.Event()

    w = InferenceWorker(IdentityModel(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    w.process_one(block=True, timeout=1.0)
    res = out_q.get_nowait()
    assert res.patient_id == "100"
    assert res.residual == pytest.approx(0.0, abs=1e-6)
    assert res.recon.shape == (720,)


def test_biased_model_yields_positive_residual():
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    w = InferenceWorker(BiasedModel(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    w.process_one(block=True, timeout=1.0)
    res = out_q.get_nowait()
    assert res.residual == pytest.approx(1.0, abs=1e-6)


def test_exception_in_forward_is_swallowed_and_logged(caplog):
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    w = InferenceWorker(Broken(), in_q, out_q, stop)
    in_q.put(StreamWindow("100", "t", np.zeros(720, dtype=float)))
    with caplog.at_level("ERROR"):
        w.process_one(block=True, timeout=1.0)
    assert out_q.empty()
    assert "inference failed" in caplog.text.lower()
```

Add `import pytest` at the top.

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_inference.py -v`
Expected: import error.

- [ ] **Step 3: Implement the worker**

`realtime/inference.py`:

```python
"""Inference worker: consumes stream windows, emits residuals."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preprocess import preprocess
from realtime.stream_source import StreamWindow

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    patient_id: str
    ts_utc: str
    raw: np.ndarray       # preprocessed input, shape (window_samples,)
    recon: np.ndarray     # reconstruction, same shape
    residual: float       # scalar MSE


def load_model(path: Path, build_fn) -> nn.Module:
    """Load trained weights into the model returned by `build_fn()`."""
    model = build_fn()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


class InferenceWorker:
    def __init__(
        self,
        model: nn.Module,
        in_queue: "queue.Queue[StreamWindow]",
        out_queue: "queue.Queue[InferenceResult]",
        stop_event: threading.Event,
        sampling_rate: int = 360,
    ) -> None:
        self._model = model
        self._in = in_queue
        self._out = out_queue
        self._stop = stop_event
        self._fs = sampling_rate

    def run(self) -> None:
        while not self._stop.is_set():
            self.process_one(block=True, timeout=0.5)

    def process_one(self, *, block: bool, timeout: float) -> None:
        try:
            win = self._in.get(block=block, timeout=timeout)
        except queue.Empty:
            return
        try:
            pre = preprocess(win.samples, fs=self._fs, apply_bandpass=True,
                             apply_notch=False)
            x = torch.from_numpy(pre).float().unsqueeze(0)
            with torch.no_grad():
                recon_t = self._model(x)
            recon = recon_t.squeeze(0).cpu().numpy().astype(float)
            residual = float(np.mean((pre - recon) ** 2))
            self._out.put(
                InferenceResult(
                    patient_id=win.patient_id,
                    ts_utc=win.ts_utc,
                    raw=pre,
                    recon=recon,
                    residual=residual,
                ),
                timeout=1.0,
            )
        except Exception as e:
            log.error("inference failed for %s: %s", win.patient_id, e)
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_inference.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/inference.py tests/test_inference.py
git commit -m "feat(realtime): inference worker computes residuals"
```

---

## Task 8: Event store — SQLite schema and writer thread

**Files:**
- Create: `realtime/event_store.py`, `tests/test_event_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_event_store.py`:

```python
import queue
import sqlite3
import threading
import time

import pytest

from realtime.event_store import (
    EventStore, AnomalyEvent, ModelVersionRecord,
)


def test_schema_created_on_init(tmp_db):
    store = EventStore(tmp_db)
    store.close()
    con = sqlite3.connect(tmp_db)
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "anomaly_events" in names
    assert "model_versions" in names


def test_log_anomaly_event_roundtrip(tmp_db):
    store = EventStore(tmp_db)
    ev = AnomalyEvent(
        patient_id="100",
        event_type="anomaly_start",
        ts_utc="2026-04-18T12:00:00+00:00",
        residual=0.9,
        threshold=0.5,
        threshold_mode="percentile",
        model_version="ecg_autoencoder.pt@abc123",
    )
    store.log_anomaly(ev)
    store.close()

    con = sqlite3.connect(tmp_db)
    rows = con.execute(
        "SELECT patient_id, event_type, residual, threshold FROM anomaly_events"
    ).fetchall()
    assert rows == [("100", "anomaly_start", 0.9, 0.5)]


def test_log_model_version(tmp_db):
    store = EventStore(tmp_db)
    mv = ModelVersionRecord(
        loaded_at_utc="2026-04-18T12:00:00+00:00",
        model_path="models/ecg_autoencoder.pt",
        model_sha256="deadbeef",
        git_commit="cafe",
        config_snapshot='{"stride_samples": 180}',
    )
    store.log_model_version(mv)
    store.close()
    con = sqlite3.connect(tmp_db)
    rows = con.execute("SELECT model_path FROM model_versions").fetchall()
    assert rows == [("models/ecg_autoencoder.pt",)]


def test_retry_on_locked_db(tmp_db, monkeypatch):
    store = EventStore(tmp_db)

    calls = {"n": 0}
    original = store._insert_anomaly

    def flaky(ev):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return original(ev)

    monkeypatch.setattr(store, "_insert_anomaly", flaky)
    ev = AnomalyEvent(
        patient_id="100", event_type="anomaly_start",
        ts_utc="t", residual=1.0, threshold=0.1,
        threshold_mode="percentile", model_version="m",
    )
    store.log_anomaly(ev)
    store.close()
    assert calls["n"] == 3


def test_writer_thread_drains_queue(tmp_db):
    store = EventStore(tmp_db)
    store.start_writer_thread()
    for i in range(5):
        store.queue_event(AnomalyEvent(
            patient_id=str(i), event_type="anomaly_start",
            ts_utc="t", residual=1.0, threshold=0.1,
            threshold_mode="percentile", model_version="m",
        ))
    store.close()  # joins thread, drains queue
    con = sqlite3.connect(tmp_db)
    count = con.execute("SELECT count(*) FROM anomaly_events").fetchone()[0]
    assert count == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_event_store.py -v`
Expected: import error.

- [ ] **Step 3: Implement EventStore**

`realtime/event_store.py`:

```python
"""SQLite-backed event log with a dedicated writer thread."""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    ts_utc          TEXT    NOT NULL,
    residual        REAL    NOT NULL,
    threshold       REAL    NOT NULL,
    threshold_mode  TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    acknowledged    INTEGER DEFAULT 0,
    ack_ts_utc      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_patient_ts
    ON anomaly_events(patient_id, ts_utc);

CREATE TABLE IF NOT EXISTS model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    loaded_at_utc   TEXT    NOT NULL,
    model_path      TEXT    NOT NULL,
    model_sha256    TEXT    NOT NULL,
    git_commit      TEXT,
    config_snapshot TEXT
);
"""

RETRY_DELAYS = (0.01, 0.05, 0.2)


@dataclass(frozen=True)
class AnomalyEvent:
    patient_id: str
    event_type: str
    ts_utc: str
    residual: float
    threshold: float
    threshold_mode: str
    model_version: str


@dataclass(frozen=True)
class ModelVersionRecord:
    loaded_at_utc: str
    model_path: str
    model_sha256: str
    git_commit: str | None
    config_snapshot: str


class EventStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self._db_path, check_same_thread=False)
        self._con.executescript(SCHEMA)
        self._con.commit()
        self._write_lock = threading.Lock()
        # chmod 600 to satisfy security scope
        try:
            self._db_path.chmod(0o600)
        except OSError:
            pass

        self._q: queue.Queue[AnomalyEvent | None] = queue.Queue()
        self._writer: threading.Thread | None = None

    def _insert_anomaly(self, ev: AnomalyEvent) -> None:
        with self._write_lock:
            self._con.execute(
                "INSERT INTO anomaly_events(patient_id, event_type, ts_utc,"
                " residual, threshold, threshold_mode, model_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ev.patient_id, ev.event_type, ev.ts_utc, ev.residual,
                 ev.threshold, ev.threshold_mode, ev.model_version),
            )
            self._con.commit()

    def log_anomaly(self, ev: AnomalyEvent) -> None:
        last_exc: Exception | None = None
        for delay in (0.0,) + RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                self._insert_anomaly(ev)
                return
            except sqlite3.OperationalError as e:
                last_exc = e
                log.warning("db busy (%s); retrying in %.3fs", e, delay)
        log.error("failed to insert anomaly event after retries: %s", last_exc)

    def log_model_version(self, mv: ModelVersionRecord) -> None:
        with self._write_lock:
            self._con.execute(
                "INSERT INTO model_versions(loaded_at_utc, model_path,"
                " model_sha256, git_commit, config_snapshot)"
                " VALUES (?, ?, ?, ?, ?)",
                (mv.loaded_at_utc, mv.model_path, mv.model_sha256,
                 mv.git_commit, mv.config_snapshot),
            )
            self._con.commit()

    # ---- writer thread plumbing ----

    def start_writer_thread(self) -> None:
        if self._writer is not None:
            return
        self._writer = threading.Thread(
            target=self._drain, name="event-store-writer", daemon=True,
        )
        self._writer.start()

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            self.log_anomaly(item)

    def queue_event(self, ev: AnomalyEvent) -> None:
        self._q.put(ev)

    def close(self) -> None:
        if self._writer is not None:
            self._q.put(None)
            self._writer.join(timeout=2.0)
            self._writer = None
        with self._write_lock:
            self._con.close()
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_event_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/event_store.py tests/test_event_store.py
git commit -m "feat(realtime): SQLite event store with writer thread and retry"
```

---

## Task 9: Log encryption utilities

**Files:**
- Create: `realtime/log_crypto.py`, `tests/test_log_crypto.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_log_crypto.py`:

```python
from pathlib import Path

from realtime.log_crypto import (
    ensure_key, encrypt_file, decrypt_file,
)


def test_key_is_generated_if_missing(tmp_path: Path):
    key_path = tmp_path / ".key"
    assert not key_path.exists()
    key = ensure_key(key_path)
    assert key_path.exists()
    assert len(key) == 44  # base64 Fernet key length


def test_existing_key_is_returned(tmp_path: Path):
    key_path = tmp_path / ".key"
    k1 = ensure_key(key_path)
    k2 = ensure_key(key_path)
    assert k1 == k2


def test_roundtrip(tmp_path: Path):
    key_path = tmp_path / ".key"
    ensure_key(key_path)
    src = tmp_path / "sample.log"
    src.write_text("hello audit trail\n")
    enc = encrypt_file(src, key_path)
    assert enc.suffix == ".enc"
    assert not src.exists()

    out = tmp_path / "restored.log"
    decrypt_file(enc, key_path, out)
    assert out.read_text() == "hello audit trail\n"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_log_crypto.py -v`
Expected: import error.

- [ ] **Step 3: Implement log crypto**

`realtime/log_crypto.py`:

```python
"""Fernet encryption for rotated log files."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet


def ensure_key(path: Path) -> bytes:
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def encrypt_file(src: Path, key_path: Path) -> Path:
    key = Path(key_path).read_bytes()
    f = Fernet(key)
    src = Path(src)
    data = src.read_bytes()
    enc_path = src.with_suffix(src.suffix + ".enc")
    enc_path.write_bytes(f.encrypt(data))
    try:
        enc_path.chmod(0o600)
    except OSError:
        pass
    src.unlink()
    return enc_path


def decrypt_file(enc_path: Path, key_path: Path, out_path: Path) -> None:
    key = Path(key_path).read_bytes()
    f = Fernet(key)
    enc_path = Path(enc_path)
    data = f.decrypt(enc_path.read_bytes())
    out_path = Path(out_path)
    out_path.write_bytes(data)
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_log_crypto.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add realtime/log_crypto.py tests/test_log_crypto.py
git commit -m "feat(realtime): Fernet encryption + decrypt utility for logs"
```

---

## Task 10: Notifier — Qt signals for the UI

**Files:**
- Create: `realtime/notifier.py`

This module is a thin glue class. It is UI-adjacent and depends on PyQt5, so it's not unit-tested directly (manual test covered in Task 15 UI checklist).

- [ ] **Step 1: Implement**

`realtime/notifier.py`:

```python
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
    event_type: str          # "anomaly_start" | "anomaly_end"
    ts_utc: str
    residual: float
    threshold: float


class PipelineSignals(QObject):
    new_window = pyqtSignal(object)          # InferenceResult
    detection = pyqtSignal(object)           # DetectionResult
    anomaly_edge = pyqtSignal(object)        # AnomalyEdge
    health = pyqtSignal(dict)                # {"queue_in": int, "p95_ms": float, ...}
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
```

- [ ] **Step 2: Smoke-import the module**

Run: `python -c "from realtime.notifier import PipelineSignals; print(PipelineSignals)"`
Expected: no error, prints the class.

- [ ] **Step 3: Commit**

```bash
git add realtime/notifier.py
git commit -m "feat(realtime): Qt signal hub for UI fan-out"
```

---

## Task 11: Pipeline — thread supervisor and queue wiring

**Files:**
- Create: `realtime/pipeline.py`, `tests/test_pipeline_headless.py`

- [ ] **Step 1: Write the failing integration-style test**

`tests/test_pipeline_headless.py`:

```python
import time

import numpy as np
import pytest
import torch
from torch import nn

from realtime.config_rt import RealtimeConfig
from realtime.event_store import EventStore
from realtime.pipeline import Pipeline


class BiasedModel(nn.Module):
    def forward(self, x):
        return x + 5.0  # guaranteed large residual


def test_pipeline_fires_anomaly_in_headless_mode(tmp_path, monkeypatch):
    # Fake record: 3-second flat signal
    class FakeRec:
        p_signal = np.zeros((3 * 360, 1), dtype=float)
        fs = 360

    import realtime.stream_source as ss
    monkeypatch.setattr(ss, "_rdrecord", lambda _p: FakeRec())

    cfg = RealtimeConfig(
        residual_buffer_w=10, warmup_min=3,
        smoother_k=1, smoother_m=1,          # instant fire
        threshold_mode="percentile", percentile_q=0.5,
        records=("100",), inference_workers=1,
    )
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    store.start_writer_thread()

    pipeline = Pipeline(cfg, model=BiasedModel(), event_store=store,
                        signals=None, headless=True, record_map={"100": "fake"})
    pipeline.start()
    # Wait up to 3 s for the fake signal to drain
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if pipeline.anomaly_count >= 1:
            break
        time.sleep(0.05)
    pipeline.stop()
    store.close()

    assert pipeline.anomaly_count >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline_headless.py -v`
Expected: import error.

- [ ] **Step 3: Implement Pipeline**

`realtime/pipeline.py`:

```python
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
from realtime.notifier import AnomalyEdge, PipelineSignals
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
        signals: Optional[PipelineSignals],
        headless: bool = False,
        record_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._store = event_store
        self._signals = signals
        self._headless = headless
        self._record_map = record_map or {r: f"data/mitbih/{r}" for r in cfg.records}

        self._stop = threading.Event()
        self._windows_q: queue.Queue[StreamWindow] = queue.Queue(maxsize=cfg.queue_maxsize)
        self._results_q: queue.Queue[InferenceResult] = queue.Queue(maxsize=cfg.queue_maxsize)

        self._detectors: dict[str, Detector] = {
            r: Detector(r, cfg) for r in cfg.records
        }
        self._threads: list[threading.Thread] = []
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
        # Record the model version on startup
        self._store.log_model_version(ModelVersionRecord(
            loaded_at_utc=_now_iso(),
            model_path=self._cfg.model_path,
            model_sha256=self._model_version.split("sha=")[1].split("@")[0],
            git_commit=self._model_version.split("git=")[1],
            config_snapshot=json.dumps(asdict(self._cfg), default=str),
        ))

        # Stream sources
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
            )
            self._spawn(f"src-{patient_id}", src.run)

        # Inference workers
        for i in range(self._cfg.inference_workers):
            w = InferenceWorker(
                self._model, self._windows_q, self._results_q, self._stop,
                sampling_rate=self._cfg.sampling_rate,
            )
            self._spawn(f"inf-{i}", w.run)

        # Detector / dispatcher
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
        ev = AnomalyEvent(
            patient_id=d.patient_id,
            event_type=event_type,
            ts_utc=result.ts_utc,
            residual=d.residual,
            threshold=d.threshold or 0.0,
            threshold_mode=self._cfg.threshold_mode,
            model_version=self._model_version,
        )
        self._store.queue_event(ev)
        if event_type == "anomaly_start":
            self.anomaly_count += 1
        if self._signals is not None:
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
```

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_pipeline_headless.py -v`
Expected: 1 passed. This may take up to 3 s.

- [ ] **Step 5: Commit**

```bash
git add realtime/pipeline.py tests/test_pipeline_headless.py
git commit -m "feat(realtime): pipeline supervisor wires streams, inference, detectors"
```

---

## Task 12: StreamPanel UI widget

**Files:**
- Create: `realtime/ui/stream_panel.py`

UI widgets are covered by the manual-test checklist (Task 16), not automated.

- [ ] **Step 1: Implement StreamPanel**

`realtime/ui/stream_panel.py`:

```python
"""Per-stream UI panel: raw + recon + residual + anomaly highlights."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from realtime.detector import DetectionResult
from realtime.inference import InferenceResult
from realtime.notifier import AnomalyEdge


BUFFER_SECONDS = 10


@dataclass
class _RingBuffer:
    capacity: int
    data: np.ndarray

    @classmethod
    def create(cls, capacity: int) -> "_RingBuffer":
        return cls(capacity=capacity, data=np.zeros(capacity, dtype=float))

    def push_chunk(self, chunk: np.ndarray) -> None:
        n = chunk.size
        if n >= self.capacity:
            self.data = chunk[-self.capacity:].copy()
        else:
            self.data = np.concatenate([self.data[n:], chunk])


class StreamPanel(QWidget):
    def __init__(self, patient_id: str, sampling_rate: int,
                 stride_samples: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._patient_id = patient_id
        self._cap = BUFFER_SECONDS * sampling_rate
        self._stride = stride_samples
        self._raw = _RingBuffer.create(self._cap)
        self._recon = _RingBuffer.create(self._cap)
        self._resid = deque(maxlen=BUFFER_SECONDS * (sampling_rate // stride_samples))
        self._threshold_val: float | None = None
        self._x = np.arange(self._cap, dtype=float) / sampling_rate
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"Patient {self._patient_id}"))
        self._status = QLabel("status: ● warmup")
        self._status.setStyleSheet("color: grey")
        header.addStretch()
        header.addWidget(self._status)
        layout.addLayout(header)

        self._raw_plot = pg.PlotWidget(title="ECG (raw vs reconstruction)")
        self._raw_curve = self._raw_plot.plot(pen=pg.mkPen("#1f77b4"))
        self._recon_curve = self._raw_plot.plot(pen=pg.mkPen("#ff7f0e"))
        self._anomaly_regions: list[pg.LinearRegionItem] = []
        layout.addWidget(self._raw_plot)

        self._res_plot = pg.PlotWidget(title="Residual and threshold")
        self._res_curve = self._res_plot.plot(pen=pg.mkPen("#555555"))
        self._thr_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("#d62728", style=Qt.DashLine)
        )
        self._res_plot.addItem(self._thr_line)
        layout.addWidget(self._res_plot)

    # ---- slots ----

    def on_window(self, r: InferenceResult) -> None:
        if r.patient_id != self._patient_id:
            return
        chunk = r.raw[-self._stride:] if r.raw.size >= self._stride else r.raw
        recon_chunk = r.recon[-self._stride:] if r.recon.size >= self._stride else r.recon
        self._raw.push_chunk(chunk)
        self._recon.push_chunk(recon_chunk)
        self._raw_curve.setData(self._x, self._raw.data)
        self._recon_curve.setData(self._x, self._recon.data)

    def on_detection(self, d: DetectionResult) -> None:
        if d.patient_id != self._patient_id:
            return
        self._resid.append(d.residual)
        self._res_curve.setData(np.arange(len(self._resid)), np.asarray(self._resid))
        if d.threshold is not None:
            self._threshold_val = d.threshold
            self._thr_line.setValue(d.threshold)
        color = {"warmup": "grey", "normal": "green", "anomaly": "red"}[d.state]
        self._status.setText(f"status: ● {d.state}")
        self._status.setStyleSheet(f"color: {color}")

    def on_edge(self, edge: AnomalyEdge) -> None:
        if edge.patient_id != self._patient_id:
            return
        if edge.event_type == "anomaly_start":
            right = self._x[-1]
            region = pg.LinearRegionItem(
                values=(right - 0.5, right),
                brush=QColor(214, 39, 40, 60),
                movable=False,
            )
            self._raw_plot.addItem(region)
            self._anomaly_regions.append(region)
```

- [ ] **Step 2: Smoke-import the module**

Run: `python -c "from realtime.ui.stream_panel import StreamPanel; print('ok')"`
Expected: prints `ok`. If PyQt/Qt platform plugin is missing set `QT_QPA_PLATFORM=offscreen`.

- [ ] **Step 3: Commit**

```bash
git add realtime/ui/stream_panel.py
git commit -m "feat(realtime): StreamPanel widget with ring buffer + pyqtgraph"
```

---

## Task 13: MainWindow and Settings dialog

**Files:**
- Create: `realtime/ui/main_window.py`, `realtime/ui/settings_dialog.py`

- [ ] **Step 1: Implement settings dialog**

`realtime/ui/settings_dialog.py`:

```python
"""Dialog for live-editing RealtimeConfig."""
from __future__ import annotations

from dataclasses import replace

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QSpinBox,
)

from realtime.config_rt import RealtimeConfig, VALID_MODES


class SettingsDialog(QDialog):
    def __init__(self, cfg: RealtimeConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Real-time settings")
        self._cfg = cfg
        form = QFormLayout(self)

        self._mode = QComboBox()
        self._mode.addItems(list(VALID_MODES))
        self._mode.setCurrentText(cfg.threshold_mode)

        self._q = QDoubleSpinBox()
        self._q.setRange(0.5, 0.999); self._q.setDecimals(3)
        self._q.setValue(cfg.percentile_q)

        self._k = QDoubleSpinBox()
        self._k.setRange(0.5, 10.0); self._k.setValue(cfg.zscore_k)

        self._sk = QSpinBox()
        self._sk.setRange(1, 10); self._sk.setValue(cfg.smoother_k)

        self._sm = QSpinBox()
        self._sm.setRange(1, 20); self._sm.setValue(cfg.smoother_m)

        self._stride = QSpinBox()
        self._stride.setRange(10, cfg.window_samples)
        self._stride.setSingleStep(10); self._stride.setValue(cfg.stride_samples)

        form.addRow("Threshold mode", self._mode)
        form.addRow("Percentile q", self._q)
        form.addRow("Z-score k", self._k)
        form.addRow("Smoother K", self._sk)
        form.addRow("Smoother M", self._sm)
        form.addRow("Stride (samples)", self._stride)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_config(self) -> RealtimeConfig:
        return replace(
            self._cfg,
            threshold_mode=self._mode.currentText(),  # type: ignore[arg-type]
            percentile_q=self._q.value(),
            zscore_k=self._k.value(),
            smoother_k=self._sk.value(),
            smoother_m=self._sm.value(),
            stride_samples=self._stride.value(),
        )
```

- [ ] **Step 2: Implement main window**

`realtime/ui/main_window.py`:

```python
"""Main application window with start/stop and one StreamPanel per patient."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QMainWindow, QPushButton, QStatusBar, QVBoxLayout, QWidget,
)

from realtime.config_rt import RealtimeConfig
from realtime.notifier import PipelineSignals
from realtime.ui.stream_panel import StreamPanel
from realtime.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(
        self,
        cfg: RealtimeConfig,
        signals: PipelineSignals,
        on_start: Callable[[RealtimeConfig], None],
        on_stop: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("ECG Anomaly Monitor")
        self._cfg = cfg
        self._signals = signals
        self._on_start = on_start
        self._on_stop = on_stop

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self._panels: dict[str, StreamPanel] = {}
        for pid in cfg.records:
            panel = StreamPanel(pid, cfg.sampling_rate, cfg.stride_samples)
            layout.addWidget(panel)
            self._panels[pid] = panel

        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._do_start)
        self._stop_btn.clicked.connect(self._do_stop)
        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)

        self.setStatusBar(QStatusBar())
        self._wire_menu()
        self._wire_signals()

    def _wire_menu(self) -> None:
        act = QAction("Settings…", self)
        act.triggered.connect(self._open_settings)
        self.menuBar().addAction(act)

    def _wire_signals(self) -> None:
        s = self._signals
        s.new_window.connect(self._fan_out_window)
        s.detection.connect(self._fan_out_detection)
        s.anomaly_edge.connect(self._fan_out_edge)

    def _fan_out_window(self, result) -> None:
        panel = self._panels.get(result.patient_id)
        if panel:
            panel.on_window(result)

    def _fan_out_detection(self, d) -> None:
        panel = self._panels.get(d.patient_id)
        if panel:
            panel.on_detection(d)

    def _fan_out_edge(self, edge) -> None:
        panel = self._panels.get(edge.patient_id)
        if panel:
            panel.on_edge(edge)
        self.statusBar().showMessage(
            f"{edge.patient_id}: {edge.event_type} at {edge.ts_utc}", 5000
        )

    def _do_start(self) -> None:
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._on_start(self._cfg)

    def _do_stop(self) -> None:
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._on_stop()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec_() == dlg.Accepted:
            self._cfg = dlg.result_config()
            self.statusBar().showMessage(
                "Settings updated. Restart pipeline to apply.", 5000
            )
```

- [ ] **Step 3: Smoke-import both modules**

Run:
```
QT_QPA_PLATFORM=offscreen python -c "
from realtime.config_rt import RealtimeConfig
from realtime.notifier import PipelineSignals
from realtime.ui.main_window import MainWindow
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add realtime/ui/main_window.py realtime/ui/settings_dialog.py
git commit -m "feat(realtime): MainWindow and SettingsDialog with Qt signal fan-out"
```

---

## Task 14: Entry point — `realtime_app.py`

**Files:**
- Create: `realtime_app.py`

- [ ] **Step 1: Implement entry point**

`realtime_app.py`:

```python
"""Entry point for the real-time ECG anomaly detection app."""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path

from realtime.config_rt import RealtimeConfig, load_config
from realtime.event_store import EventStore
from realtime.inference import load_model
from realtime.log_crypto import ensure_key, encrypt_file
from realtime.pipeline import Pipeline
from models import ConvAutoencoder

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "system.log"
DB_FILE = LOG_DIR / "events.db"
KEY_FILE = LOG_DIR / ".key"


def _install_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    ensure_key(KEY_FILE)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    original_do_rollover = handler.doRollover

    def _rollover_and_encrypt():
        original_do_rollover()
        # Encrypt the most-recently-rotated file (system.log.1)
        rotated = LOG_FILE.with_suffix(LOG_FILE.suffix + ".1")
        if rotated.exists():
            try:
                encrypt_file(rotated, KEY_FILE)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "log rotation encrypt failed: %s", e
                )

    handler.doRollover = _rollover_and_encrypt  # type: ignore[assignment]

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stderr))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--records", type=str, default=None,
                   help="Comma-separated record IDs, overrides config")
    p.add_argument("--seconds", type=float, default=0,
                   help="Headless only: stop after N seconds")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> RealtimeConfig:
    cfg = load_config(args.config) if args.config else RealtimeConfig()
    if args.records:
        cfg = cfg.__class__(**{**cfg.__dict__,
                               "records": tuple(args.records.split(","))})
    return cfg


def _run_headless(cfg: RealtimeConfig, seconds: float) -> int:
    import time

    store = EventStore(DB_FILE)
    store.start_writer_thread()
    model = load_model(Path(cfg.model_path), ConvAutoencoder)
    pipeline = Pipeline(cfg, model=model, event_store=store,
                        signals=None, headless=True)
    pipeline.start()
    try:
        if seconds > 0:
            time.sleep(seconds)
        else:
            print("press Ctrl+C to stop")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        store.close()
    print(f"Captured {pipeline.anomaly_count} anomaly starts")
    return 0


def _run_gui(cfg: RealtimeConfig) -> int:
    from PyQt5.QtWidgets import QApplication
    from realtime.notifier import PipelineSignals
    from realtime.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    store = EventStore(DB_FILE)
    store.start_writer_thread()
    model = load_model(Path(cfg.model_path), ConvAutoencoder)
    signals = PipelineSignals()
    pipeline: Pipeline | None = None

    def on_start(current_cfg: RealtimeConfig) -> None:
        nonlocal pipeline
        pipeline = Pipeline(current_cfg, model=model, event_store=store,
                            signals=signals, headless=False)
        pipeline.start()

    def on_stop() -> None:
        nonlocal pipeline
        if pipeline is not None:
            pipeline.stop()
            pipeline = None

    window = MainWindow(cfg, signals, on_start=on_start, on_stop=on_stop)
    window.resize(1000, 800)
    window.show()
    try:
        return app.exec_()
    finally:
        on_stop()
        store.close()


def main() -> int:
    _install_logging()
    args = _parse_args()
    cfg = _build_config(args)
    logging.info("loaded config: %s", json.dumps(cfg.__dict__, default=str))
    if args.headless:
        return _run_headless(cfg, args.seconds)
    return _run_gui(cfg)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test headless import**

Run: `python -c "import realtime_app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Headless dry run against a real MIT-BIH record**

Run (requires a prior `python dataset.py --download` that produced `data/mitbih/`):
```
python realtime_app.py --headless --records 208 --seconds 20
```
Expected: logs startup, writes rows to `logs/events.db`, prints "Captured N anomaly starts" where N ≥ 1. If the command fails because data is missing, skip this step and note in the commit message that it requires the dataset.

- [ ] **Step 4: Commit**

```bash
git add realtime_app.py
git commit -m "feat(realtime): entry point with GUI and --headless modes"
```

---

## Task 15: Integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the test**

This test is gated behind a marker so it only runs when MIT-BIH data is present.

`tests/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run locally if data exists**

Run: `pytest tests/test_integration.py -v`
Expected: passes if MIT-BIH data is present; skipped otherwise (both outcomes are acceptable).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(realtime): headless integration test on record 208"
```

---

## Task 16: Latency benchmark

**Files:**
- Create: `tests/test_latency.py`

- [ ] **Step 1: Write the gated benchmark**

`tests/test_latency.py`:

```python
import os
import queue
import threading
import time

import numpy as np
import pytest
import torch

from realtime.inference import InferenceResult, InferenceWorker, load_model
from realtime.stream_source import StreamWindow
from models import ConvAutoencoder


pytestmark = pytest.mark.latency


@pytest.mark.skipif(
    os.environ.get("TORCH_CPU_BENCHMARK") != "1",
    reason="set TORCH_CPU_BENCHMARK=1 to run the latency benchmark",
)
def test_p95_latency_under_100ms():
    from pathlib import Path
    model = load_model(Path("models/ecg_autoencoder.pt"), ConvAutoencoder)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    worker = InferenceWorker(model, in_q, out_q, stop, sampling_rate=360)

    N = 100
    for _ in range(N):
        in_q.put(StreamWindow("100", "t", np.random.randn(720).astype(float)))

    latencies = []
    for _ in range(N):
        start = time.perf_counter()
        worker.process_one(block=True, timeout=5.0)
        latencies.append((time.perf_counter() - start) * 1000)

    p95 = float(np.quantile(latencies, 0.95))
    print(f"inference p95 = {p95:.1f} ms")
    assert p95 < 100.0, f"p95 {p95:.1f} ms exceeds 100 ms budget"
```

- [ ] **Step 2: Run locally**

Run: `TORCH_CPU_BENCHMARK=1 pytest tests/test_latency.py -v -s`
Expected: passes with p95 well under 100 ms on a laptop. If skipped, CI contract is still satisfied.

- [ ] **Step 3: Commit**

```bash
git add tests/test_latency.py
git commit -m "test(realtime): gated latency benchmark (p95 < 100ms)"
```

---

## Task 17: Docker + docker-compose

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `docs/running-in-docker.md`

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libxcb1 libxkbcommon0 libx11-6 libfontconfig1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p logs data

# Default to headless for container use; GUI is a separate launch target
CMD ["python", "realtime_app.py", "--headless", "--records", "208"]
```

- [ ] **Step 2: docker-compose.yml**

```yaml
services:
  ecg-rt:
    build: .
    container_name: ecg-rt
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - TORCH_NUM_THREADS=2
    restart: unless-stopped
```

- [ ] **Step 3: docs/running-in-docker.md**

```markdown
# Running the ECG real-time pipeline in Docker

## Headless mode (any platform)

```bash
docker compose build
docker compose up
# Logs and SQLite event database persist in ./logs
```

## GUI mode

**Linux:**
```bash
xhost +local:docker
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ecg-rt python realtime_app.py
```

**macOS / Windows:** GUI-in-Docker is not officially supported in this
project. Run the GUI natively and use Docker only for headless evaluation
and CI. This is called out in the thesis as a known limitation.
```

- [ ] **Step 4: Build to verify**

Run: `docker build -t ecg-rt .`
Expected: completes successfully. If Docker is not available locally, note this in the commit message — the files are still committed.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml docs/running-in-docker.md
git commit -m "build(realtime): Docker image + compose file (headless default)"
```

---

## Task 18: Manual test checklist

**Files:**
- Create: `docs/manual-test.md`

- [ ] **Step 1: Author the checklist**

```markdown
# Manual test checklist for the real-time UI

Run before a thesis demo or release candidate.

## Precondition

- [ ] MIT-BIH records 100, 208, 222 available under `data/mitbih/`
- [ ] Trained model at `models/ecg_autoencoder.pt`
- [ ] `pip install -r requirements.txt` completed in the active venv

## Launch

- [ ] `python realtime_app.py` opens a main window with 3 patient panels
- [ ] Status dots read "● warmup" (grey) for all three
- [ ] Clicking **Start** kicks the pipeline off — within ~3 seconds the raw
      (blue) and reconstruction (orange) traces begin scrolling

## Detection

- [ ] After ~30 s, record 208's panel flips to "● anomaly" (red) at least once
- [ ] A red region shades the raw-ECG plot at the time of detection
- [ ] The status-bar toast reads "208: anomaly_start at <timestamp>"
- [ ] `sqlite3 logs/events.db "select count(*) from anomaly_events"` returns ≥ 1

## Threshold live-switch

- [ ] Open Settings, change mode to `zscore`, press OK
- [ ] Status-bar toast says "Settings updated. Restart pipeline to apply."
- [ ] Click Stop, then Start — dashed threshold line visibly changes shape

## Graceful shutdown

- [ ] Click Stop: status dots return to "● normal" within 1–2 s, plots freeze
- [ ] Close the window: `logs/events.db` is intact (no corruption)
- [ ] Re-open and Start again: new events append to existing DB

## Docker headless

- [ ] `docker compose up` runs for 30 s, produces logs and DB rows
```

- [ ] **Step 2: Commit**

```bash
git add docs/manual-test.md
git commit -m "docs(realtime): manual-test checklist for UI smoke tests"
```

---

## Task 19: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run: `cat README.md | head -30`. Confirm layout before editing.

- [ ] **Step 2: Append a "Real-time pipeline" section**

Insert the following block immediately before the "## License" section:

```markdown
## Real-time pipeline

The `realtime/` package adds a live pipeline that replays MIT-BIH records at
360 Hz, detects anomalies with dynamic thresholding and N-of-M smoothing, and
visualises 2–3 concurrent streams in a PyQt desktop UI.

### Headless demo (CI-friendly)

```bash
python realtime_app.py --headless --records 208 --seconds 30
```

### GUI mode

```bash
python realtime_app.py
```

### Configuration

Defaults live in `realtime/config_rt.py` (`RealtimeConfig` dataclass).
To override, write a JSON file and pass it with `--config`:

```bash
python realtime_app.py --config my_config.json
```

### Architecture

Documented in `docs/superpowers/specs/2026-04-18-realtime-ecg-pipeline-design.md`.
Each module in `realtime/` maps to a thesis Section 3.1.* subsection.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document real-time pipeline usage"
```

---

## Post-implementation checklist

After all 19 tasks are complete:

- [ ] `pytest -q` — all unit tests pass
- [ ] `QT_QPA_PLATFORM=offscreen python -c "from realtime.ui.main_window import MainWindow; print('ok')"` — UI imports cleanly
- [ ] `python realtime_app.py --headless --records 208 --seconds 30` — ≥ 1 anomaly captured
- [ ] Manual checklist at `docs/manual-test.md` walked through at least once
- [ ] `TORCH_CPU_BENCHMARK=1 pytest tests/test_latency.py` — p95 < 100 ms
- [ ] `docker build -t ecg-rt .` succeeds
- [ ] Spec document and this plan cross-referenced in the thesis Implementation chapter

When complete, invoke `superpowers:finishing-a-development-branch` to decide how to integrate the work.
