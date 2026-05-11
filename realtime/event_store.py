"""SQLite-backed event log with a dedicated writer thread."""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id             TEXT    NOT NULL,
    event_type             TEXT    NOT NULL,
    ts_utc                 TEXT    NOT NULL,
    record_offset_seconds  REAL    DEFAULT NULL,
    residual               REAL    NOT NULL,
    threshold              REAL    NOT NULL,
    threshold_mode         TEXT    NOT NULL,
    model_version          TEXT    NOT NULL,
    acknowledged           INTEGER DEFAULT 0,
    ack_ts_utc             TEXT,
    waveform               BLOB    DEFAULT NULL,
    reconstruction         BLOB    DEFAULT NULL,
    peak_sample_index      INTEGER DEFAULT NULL,
    clinician_label        TEXT    DEFAULT NULL,
    clinician_notes        TEXT    DEFAULT NULL,
    reviewed_ts_utc        TEXT    DEFAULT NULL,
    reviewer               TEXT    DEFAULT NULL
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

# Idempotent ALTERs to upgrade older databases that pre-date the new columns.
# SQLite has no IF NOT EXISTS for ADD COLUMN, so we attempt and swallow the
# duplicate-column error.
_MIGRATIONS = [
    "ALTER TABLE anomaly_events ADD COLUMN waveform BLOB DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN reconstruction BLOB DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN peak_sample_index INTEGER DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN clinician_label TEXT DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN clinician_notes TEXT DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN reviewed_ts_utc TEXT DEFAULT NULL",
    "ALTER TABLE anomaly_events ADD COLUMN reviewer TEXT DEFAULT NULL",
]


def _apply_migrations(con: sqlite3.Connection) -> None:
    for stmt in _MIGRATIONS:
        try:
            con.execute(stmt)
        except sqlite3.OperationalError as e:
            # "duplicate column name" means migration already applied
            if "duplicate column name" not in str(e).lower():
                raise
    con.commit()

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
    record_offset_seconds: float | None = None
    # Optional payload populated for "anomaly_start" events so the
    # cardiologist can review the exact window that fired the alert.
    waveform: np.ndarray | None = None       # 720-sample preprocessed window
    reconstruction: np.ndarray | None = None # 720-sample model output (same scale)
    peak_sample_index: int | None = None     # arg-max of |x - x_recon|² in [0, 720)


@dataclass(frozen=True)
class StoredAnomaly:
    """Read-side projection used by the review dialog."""
    id: int
    patient_id: str
    event_type: str
    ts_utc: str
    record_offset_seconds: float | None
    residual: float
    threshold: float
    threshold_mode: str
    model_version: str
    waveform: np.ndarray | None
    reconstruction: np.ndarray | None
    peak_sample_index: int | None
    clinician_label: str | None
    clinician_notes: str | None
    reviewed_ts_utc: str | None
    reviewer: str | None


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
        _apply_migrations(self._con)
        self._write_lock = threading.Lock()
        try:
            self._db_path.chmod(0o600)
        except OSError:
            pass

        self._q: queue.Queue[AnomalyEvent | None] = queue.Queue()
        self._writer: threading.Thread | None = None

    def _insert_anomaly(self, ev: AnomalyEvent) -> None:
        wf_blob = (
            np.ascontiguousarray(ev.waveform, dtype=np.float32).tobytes()
            if ev.waveform is not None else None
        )
        recon_blob = (
            np.ascontiguousarray(ev.reconstruction, dtype=np.float32).tobytes()
            if ev.reconstruction is not None else None
        )
        with self._write_lock:
            self._con.execute(
                "INSERT INTO anomaly_events(patient_id, event_type, ts_utc,"
                " record_offset_seconds, residual, threshold, threshold_mode,"
                " model_version, waveform, reconstruction, peak_sample_index)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ev.patient_id, ev.event_type, ev.ts_utc,
                 ev.record_offset_seconds,
                 ev.residual, ev.threshold, ev.threshold_mode, ev.model_version,
                 wf_blob, recon_blob, ev.peak_sample_index),
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

    _SELECT_COLS = (
        "id, patient_id, event_type, ts_utc, record_offset_seconds,"
        " residual, threshold, threshold_mode, model_version,"
        " waveform, reconstruction, peak_sample_index,"
        " clinician_label, clinician_notes,"
        " reviewed_ts_utc, reviewer"
    )

    def list_anomaly_starts(
        self,
        *,
        patient_id: str | None = None,
        limit: int = 200,
    ) -> list[StoredAnomaly]:
        """Return the most-recent anomaly_start rows (newest first)."""
        sql = (f"SELECT {self._SELECT_COLS} FROM anomaly_events"
               " WHERE event_type = 'anomaly_start'")
        params: list = []
        if patient_id is not None:
            sql += " AND patient_id = ?"
            params.append(patient_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._write_lock:
            cur = self._con.execute(sql, params)
            rows = cur.fetchall()
        return [_row_to_stored(r) for r in rows]

    def get_anomaly(self, event_id: int) -> StoredAnomaly | None:
        sql = (f"SELECT {self._SELECT_COLS} FROM anomaly_events WHERE id = ?")
        with self._write_lock:
            cur = self._con.execute(sql, (int(event_id),))
            row = cur.fetchone()
        return _row_to_stored(row) if row is not None else None

    def set_clinician_decision(
        self,
        event_id: int,
        *,
        label: str,
        notes: str | None = None,
        reviewed_ts_utc: str | None = None,
        reviewer: str | None = None,
    ) -> None:
        """Record the cardiologist's TP / FP / UNSURE adjudication of an alert.

        ``reviewer`` is a free-form text identifier for the clinician (name,
        hospital ID, or initials). Stored verbatim alongside the decision so
        that multi-rater kappa replication can attribute each label.
        """
        if label not in {"TP", "FP", "UNSURE"}:
            raise ValueError(f"label must be one of TP/FP/UNSURE, got {label!r}")
        with self._write_lock:
            self._con.execute(
                "UPDATE anomaly_events"
                " SET clinician_label = ?, clinician_notes = ?,"
                " reviewed_ts_utc = ?, acknowledged = 1, ack_ts_utc = ?,"
                " reviewer = ?"
                " WHERE id = ?",
                (label, notes, reviewed_ts_utc, reviewed_ts_utc, reviewer,
                 int(event_id)),
            )
            self._con.commit()

    def update_notes(
        self,
        event_id: int,
        *,
        notes: str | None,
        reviewer: str | None = None,
    ) -> None:
        """Save free-text notes (and optionally the reviewer) WITHOUT
        changing the TP/FP/UNSURE decision. Use when the cardiologist wants
        to record an observation but is not yet ready to adjudicate."""
        with self._write_lock:
            self._con.execute(
                "UPDATE anomaly_events SET clinician_notes = ?,"
                " reviewer = COALESCE(?, reviewer)"
                " WHERE id = ?",
                (notes, reviewer, int(event_id)),
            )
            self._con.commit()

    def close(self) -> None:
        if self._writer is not None:
            self._q.put(None)
            self._writer.join(timeout=2.0)
            self._writer = None
        with self._write_lock:
            self._con.close()


def _row_to_stored(row: tuple) -> StoredAnomaly:
    (eid, pid, etype, ts, off, resid, thr, mode, mv,
     wf_blob, recon_blob, peak_idx,
     label, notes, reviewed_ts, reviewer) = row
    waveform = (
        np.frombuffer(wf_blob, dtype=np.float32).copy()
        if wf_blob is not None else None
    )
    reconstruction = (
        np.frombuffer(recon_blob, dtype=np.float32).copy()
        if recon_blob is not None else None
    )
    return StoredAnomaly(
        id=int(eid), patient_id=pid, event_type=etype, ts_utc=ts,
        record_offset_seconds=off, residual=resid, threshold=thr,
        threshold_mode=mode, model_version=mv,
        waveform=waveform, reconstruction=reconstruction,
        peak_sample_index=peak_idx,
        clinician_label=label, clinician_notes=notes,
        reviewed_ts_utc=reviewed_ts, reviewer=reviewer,
    )
