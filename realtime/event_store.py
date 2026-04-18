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
