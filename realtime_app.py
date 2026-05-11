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
        rotated = LOG_FILE.with_suffix(LOG_FILE.suffix + ".1")
        if rotated.exists():
            try:
                encrypt_file(rotated, KEY_FILE)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "log rotation encrypt failed: %s", e
                )

    handler.doRollover = _rollover_and_encrypt

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
    pipeline = None

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

    window = MainWindow(cfg, signals, on_start=on_start, on_stop=on_stop,
                        event_store=store)
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
