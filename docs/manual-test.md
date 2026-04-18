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
