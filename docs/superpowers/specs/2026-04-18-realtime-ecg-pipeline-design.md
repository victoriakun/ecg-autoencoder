# Real-time ECG Anomaly Detection Pipeline — Design

**Status:** Approved via brainstorming, pending written-spec review
**Date:** 2026-04-18
**Author:** Viktoria Kun (with Claude)
**Scope:** Implements Sections 3.1.1 through 3.1.6, 3.1.8, and the medium subset of 3.2 from the thesis requirements.

## 1. Goal

Extend the existing batch-trained convolutional autoencoder project into a live system that replays MIT-BIH ECG records at 360 Hz, detects anomalies in near real time using dynamic thresholding, and visualises them in a desktop UI — producing a demonstrable thesis artefact with a defensible evaluation chapter.

The current repository has `train_mitbih.py`, `evaluate.py`, and a trained model (`models/ecg_autoencoder.pt`, ROC-AUC 97.21 %, F1 85.14 %). What it lacks is any live pipeline: the user runs evaluation offline against a pre-built dataset. This design adds that live pipeline without disturbing the batch code paths.

## 2. Non-goals

The following items from Section 3.2 are **not implemented** and will be discussed as future work in the thesis:

- User authentication, RBAC, HTTPS, TLS certificates
- Multi-node / Kubernetes deployment, high availability
- HIPAA audit-grade logging, GDPR data-subject workflows beyond pseudonymisation
- Offline-then-sync operation across machines
- Full UI-in-Docker support on macOS/Windows (Docker is provided for the headless mode; native run is the UI-supported path)

## 3. Scope anchors (decisions made in brainstorming)

| Decision | Choice |
|---|---|
| Input source | Replay MIT-BIH records at 360 Hz (simulated stream) |
| UI framework | PyQt + pyqtgraph |
| Non-functional scope | Medium: pipeline + UI + logging + Docker + basic log encryption |
| Model | Main conv autoencoder (`models/ecg_autoencoder.pt`) |
| Threshold | Configurable: rolling percentile (default), z-score, fixed+online |
| Concurrency | 2–3 concurrent streams via thread pool |
| Logging | SQLite for anomaly events + Python `logging` for system health |
| Anomaly smoothing | N-of-M rule (default 2-of-3) |
| Window stride | 0.5 s (75 % overlap on 2-s windows) |
| Architecture | Threaded producer/consumer with Qt signals |

## 4. Component architecture

Seven new or extended modules, one per thesis functional subsection:

| Module | File | Thesis link | Responsibility |
|---|---|---|---|
| Stream source | `stream_source.py` (new) | 3.1.1 | Replay MIT-BIH record at 360 Hz; emit `(patient_id, timestamp, samples)` windows; drop corrupt/short windows with WARNING |
| Preprocessor | `preprocess.py` (extend) | 3.1.2 | Reuse training-time bandpass 0.5–40 Hz + amplitude normalisation; single function reused by batch and streaming code paths |
| Inference | `inference.py` (new) | 3.1.3 | Load `ecg_autoencoder.pt`, forward pass, return `(recon, residual)` |
| Detector | `detector.py` (new) | 3.1.4 | Per-stream rolling residual buffer, dynamic threshold (3 modes), N-of-M smoother, rising/falling edge events |
| Event store | `event_store.py` (new) | 3.1.5 | Dedicated thread drains an event queue → SQLite; Python `logging` for system health |
| Notifier | `notifier.py` (new) | 3.1.5 | Qt signals to UI slots + optional OS desktop notification |
| UI | `ui/main_window.py`, `ui/stream_panel.py` (new) | 3.1.6 | Main window with one `StreamPanel` per active patient |
| Entry point | `realtime_app.py` (new) | — | Wires modules, owns the Qt event loop, supports `--headless` for CI/eval |

`config.py` is extended with a `RealtimeConfig` dataclass — stride, threshold mode and parameters, smoother K/M, window maxlen, number of streams, model path. Existing batch-mode constants remain untouched.

## 5. Threading model & data flow

```
[StreamSource thread × N]  ──►  windows_queue (bounded, maxsize=8)
                                       │
                                       ▼
                          [InferenceWorker thread × 2]
                              · preprocess(window)
                              · model.forward(x)
                              · residual = mse(x, recon)
                                       │
                                       ▼
                                results_queue
                                       │
                        ┌──────────────┼──────────────────┐
                        ▼              ▼                  ▼
               [Detector thread]  [UI main thread]  [EventStore thread]
               (per-stream)        (pyqtgraph)       (SQLite writer)
```

Design notes:

- **Bounded queues.** `maxsize=8` on both queues; `put()` blocks briefly if the consumer is behind, giving the system back-pressure. A sustained full queue triggers a `WARNING` in system health logs (satisfies 3.2.1's self-monitoring).
- **Shared inference pool.** One thread pool serves all streams rather than one pool per stream. PyTorch releases the GIL during forward pass, so a pool size of 2 is sufficient for 2–3 streams on a laptop CPU while avoiding loading multiple copies of the model into memory.
- **Per-stream detector.** Each stream has its own `Detector` instance holding the rolling residual buffer and N-of-M decision buffer. The `Detector` thread dispatches results to the right detector based on `patient_id`.
- **Qt signal-slot UI updates.** The UI receives two Qt signals: (1) `new_window(patient_id, raw, recon, residual)` emitted by the `Detector` once it has annotated the result with the current threshold — this feeds the waveform and residual plots; (2) `anomaly_state_changed(patient_id, event_type, ts, residual, threshold)` emitted on rising/falling edges — this drives the status dot, red shading, and alert strip. Both signals are received on the GUI thread's slots, so no cross-thread widget access.
- **Graceful shutdown.** A shared `threading.Event` is polled at every `queue.get(timeout=…)` call. On stop: sources stop, queues drain, event store flushes, window closes.

### Latency budget

2-s window, 0.5-s stride → one window per stream every 500 ms.

| Stage | Budget |
|---|---|
| Queue wait + preprocess | ~10 ms |
| Inference (CPU, ~580 k params) | ~20 ms |
| Detector + UI paint | ~10 ms |
| **End-to-end** | **~50 ms** (10× headroom vs 1 s NFR) |

## 6. Window strategy

- **Window size:** 720 samples = 2 s at 360 Hz (matches training).
- **Stride (default):** 180 samples = 0.5 s (75 % overlap). Rationale in thesis: enables sub-second anomaly localisation and gives the N-of-M smoother ~1.5 s to confirm.
- **Training was non-overlapping.** Overlap is an inference-mode choice; reconstruction is local, so the model is stride-agnostic. This is documented explicitly in the thesis Implementation chapter.
- **Configurable** via `RealtimeConfig.stride_samples` so stride can be swept in the evaluation chapter.

## 7. Dynamic threshold & smoothing

Per-stream state inside `Detector`:

```python
residual_buffer: deque[float]     # maxlen = W (default 600 = 5 min at 0.5 s stride)
decisions:       deque[bool]      # maxlen = M (default 3)
state:           "normal" | "anomaly"   # updated on edge transitions
```

Warm-up: threshold is not computed until `len(residual_buffer) >= 60`. Before that, decisions are forced to `False`.

**Three threshold modes:**

| Mode | Formula | Default params | Strengths | Weaknesses |
|---|---|---|---|---|
| `percentile` (default) | `np.quantile(buf, q)` | `q = 0.99` | Robust; patient-adaptive | Slight lag on abrupt distribution shifts |
| `zscore` | `mean(buf) + k·std(buf)` | `k = 3` | Classic, explainable | Vulnerable to being "poisoned" when anomalies fill the buffer |
| `fixed_online` | Start from training-set residual p99, nudge ±5 % via recent median | p99 from calibration | Most stable | Least adaptive |

For `fixed_online`, the starting threshold is computed once from the training residual distribution and saved to `models/ecg_autoencoder.calibration.json` next to the `.pt` file.

**N-of-M smoother:**

```python
decisions.append(residual > threshold)
confirmed = sum(decisions) >= K          # default K=2, M=3
```

On rising edge (`state == normal` → `confirmed`), detector emits `anomaly_start`. On falling edge (`confirmed → normal`), detector emits `anomaly_end`. This prevents alert spam and satisfies the thesis "smoothed or confirmed" requirement.

Every emitted event carries the current residual, threshold, threshold mode, buffer fill, and K-of-M state — a rich audit trail for the evaluation chapter.

## 8. Logging & persistence

### SQLite schema (`logs/events.db`)

```sql
CREATE TABLE anomaly_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,     -- 'anomaly_start' | 'anomaly_end'
    ts_utc          TEXT    NOT NULL,     -- ISO-8601
    residual        REAL    NOT NULL,
    threshold       REAL    NOT NULL,
    threshold_mode  TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,     -- model filename + git sha
    acknowledged    INTEGER DEFAULT 0,    -- 0/1, set by UI on click
    ack_ts_utc      TEXT
);
CREATE INDEX idx_events_patient_ts ON anomaly_events(patient_id, ts_utc);

CREATE TABLE model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    loaded_at_utc   TEXT    NOT NULL,
    model_path      TEXT    NOT NULL,
    model_sha256    TEXT    NOT NULL,
    git_commit      TEXT,
    config_snapshot TEXT                   -- JSON of RealtimeConfig at load time
);
```

`model_versions` receives a row on every startup — satisfies 3.2.5's "log of model versions."

### System-health log

Python `logging` writes to `logs/system.log` via `RotatingFileHandler` (10 MB × 5 files).

- `INFO` — stream start/stop, threshold recomputes, config load
- `WARNING` — queue backlog sustained > 5 s, corrupt/missing record, inference skip
- `ERROR` — model load failure, DB lock after 3 retries, worker-thread exception

### Log encryption (medium scope)

On startup: if `logs/.key` does not exist, generate a Fernet key and write it with `chmod 600`. A log rotation hook encrypts rotated files (`system.log.1` → `system.log.1.enc`) and deletes the plaintext. The live `system.log` stays plaintext so `tail -f` works during demos. A `decrypt_logs.py` utility reads them back. The thesis notes this as a pragmatic baseline, not clinical-grade.

## 9. UI layout

One PyQt main window. Vertical stack of `StreamPanel` widgets, one per active patient (2–3 panels visible).

```
┌──────────────────────────────────────────────────────────┐
│  ECG Anomaly Monitor                      [Start] [Stop] │
├──────────────────────────────────────────────────────────┤
│  Patient 100                    status: ● normal         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  raw ECG (blue) + reconstruction (orange)          │  │
│  │  ────scrolling 10-second window────                │  │
│  │  anomalous regions shaded red                      │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  residual (grey) + dynamic threshold (dashed red)  │  │
│  └────────────────────────────────────────────────────┘  │
│  Recent alerts: [07:12:31 anomaly ✓ ack] [07:12:48 …]    │
├──────────────────────────────────────────────────────────┤
│  Patient 208 …                                           │
└──────────────────────────────────────────────────────────┘
```

### Per-panel widgets

- `pyqtgraph.PlotWidget` #1 — raw + reconstruction overlay; 10-s rolling view; `LinearRegionItem`s (red, semi-transparent) shade confirmed anomaly spans
- `pyqtgraph.PlotWidget` #2 — residual trace with a dashed horizontal line for the current threshold (redraws as threshold adapts)
- Status indicator dot (green = normal, red = anomaly, grey = warm-up), bound to detector state
- Alert strip showing last 5 events from `event_store`; clicking acknowledges (writes `acknowledged=1`)

### Global controls

- `Start` / `Stop` buttons (start/stop all streams)
- Record-selector dropdown: pick which MIT-BIH records to replay
- `Settings…` dialog bound to `RealtimeConfig` — threshold mode, K, M, stride, `q`/`k` per mode. Live-switchable so they can be changed during the thesis defence.

### Performance approach

Each channel keeps a fixed-size NumPy ring buffer (10 s × 360 Hz = 3 600 samples). On each `new_result` Qt signal: update the ring buffer in place, then call `plot.setData(ring_buffer)` once per plot. pyqtgraph's fast path.

## 10. Docker & deployment

- **Base image:** `python:3.11-slim`
- **Dockerfile** installs system deps for Qt (`libxcb`, `libx11`, `libgl1`), pip-installs from `requirements.txt`, copies code + `models/ecg_autoencoder.pt`
- **docker-compose.yml** with one service, volumes for `logs/` and `data/`
- **UI passthrough** documented in `docs/running-in-docker.md` (Linux: `DISPLAY` + X socket mount; macOS: XQuartz)
- **Headless mode:** `realtime_app.py --headless --records 100,208` runs without the UI, writes events to SQLite, exits 0 on clean shutdown. Used by CI and the latency benchmark.

## 11. Reliability & error handling

- Each worker thread wrapped in a `try/except Exception:` loop; on exception, log at `ERROR` and continue. A supervisor restarts a crashed thread.
- SQLite writes retry on `OperationalError` (locked DB) up to 3 times with exponential backoff (10 ms, 50 ms, 200 ms).
- `StreamSource` drops windows that contain NaN or are shorter than `WINDOW_SAMPLES`, emitting a `WARNING` (matches 3.1.1's "skip or warn").
- Health metrics exposed in the UI status bar: queue depths, dropped-window count, inference p95 latency.

## 12. Testing

### Unit tests (`tests/`, pytest)

- `test_detector.py` — synthetic residual sequences feed each threshold mode; assert threshold values and rising/falling-edge events
  - percentile: flat buffer + spike → fires once
  - zscore: rising baseline → threshold drifts up (documents the "poisoning" weakness)
  - N-of-M: 1-of-3 stays silent; 2-of-3 fires; correct de-bouncing on resolved
- `test_stream_source.py` — mock `wfdb`; verify windows emitted at expected rate ± 10 %; corruption handling
- `test_event_store.py` — insert + query + retry path on simulated DB lock
- `test_preprocess.py` — known-input regression against a saved reference (guards against accidental divergence from training preprocessing)

### Integration test (`tests/test_integration.py`)

- Replay a 30-second slice of MIT-BIH record 208 through the full pipeline in headless mode
- Assert: ≥ 1 anomaly event written to SQLite; no `ERROR`-level logs; pipeline shuts down cleanly
- Runs in < 5 s via a `FastClock` injection (test-only shortcut for `StreamSource` to emit without wall-clock sleeps)

### Performance regression (`tests/test_latency.py`)

- Run 100 windows through the inference worker; measure p95 latency
- Assert p95 < 100 ms (10× headroom vs 1 s NFR)
- Skipped on CI unless `TORCH_CPU_BENCHMARK=1`; run locally before milestones

### Not automated

- PyQt UI interactions (would need `pytest-qt` and adds CI fragility). Covered by `docs/manual-test.md` checklist.

## 13. Thesis-chapter mapping

This spec is intentionally organised so each section maps to a thesis subsection:

| Thesis § | Spec section |
|---|---|
| 3.1.1 ECG database | 4 (StreamSource), 11 (corrupt-data handling) |
| 3.1.2 Preprocessor | 4 (Preprocessor), 6 (Window strategy) |
| 3.1.3 Automatic coding model | 4 (Inference), 5 (Latency budget) |
| 3.1.4 Anomaly detector | 7 (Threshold & smoothing) |
| 3.1.5 Notification and logging | 8 (Logging & persistence) |
| 3.1.6 User interface | 9 (UI layout) |
| 3.1.8 Determination of metrics | 12 (Tests), plus existing `evaluate.py` |
| 3.2.1 Performance and latency | 5 (Latency budget), 11, 12 (latency test) |
| 3.2.2 Reliability | 11 (Reliability) |
| 3.2.3 Scalability | 5 (shared pool), explicitly bounded at 2–3 streams, N-stream listed as future work |
| 3.2.4 Security | 8 (log encryption), pseudonymisation via MIT-BIH IDs, remainder in non-goals |
| 3.2.5 Maintainability | 4 (module split), 8 (`model_versions` table) |
| 3.2.6 Portability | 10 (Docker) |

## 14. Open questions for written-spec review

None at this time. If the user flags a section during review, it will be added here.
