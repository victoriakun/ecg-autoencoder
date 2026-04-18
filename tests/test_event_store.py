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
    store.close()
    con = sqlite3.connect(tmp_db)
    count = con.execute("SELECT count(*) FROM anomaly_events").fetchone()[0]
    assert count == 5
