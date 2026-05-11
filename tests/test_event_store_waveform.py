"""Schema-migration + waveform round-trip tests for the EventStore."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from realtime.event_store import (
    AnomalyEvent, EventStore, StoredAnomaly, _apply_migrations,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "events.db"


def _make_event(patient_id: str = "100", *, with_waveform: bool = True) -> AnomalyEvent:
    rng = np.random.default_rng(0)
    wf = rng.standard_normal(720).astype(np.float32) if with_waveform else None
    rec = (wf + 0.05 * rng.standard_normal(720).astype(np.float32)
           if wf is not None else None)
    peak = int(np.abs(wf).argmax()) if wf is not None else None
    return AnomalyEvent(
        patient_id=patient_id, event_type="anomaly_start",
        ts_utc="2026-05-02T10:00:00Z",
        residual=0.123, threshold=0.043, threshold_mode="percentile",
        model_version="test@sha=abc", record_offset_seconds=42.5,
        waveform=wf, reconstruction=rec, peak_sample_index=peak,
    )


def test_new_schema_includes_review_columns(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    cur = store._con.execute("PRAGMA table_info(anomaly_events)")
    cols = {row[1] for row in cur.fetchall()}
    expected = {"waveform", "reconstruction", "peak_sample_index",
                "clinician_label", "clinician_notes", "reviewed_ts_utc",
                "reviewer"}
    assert expected.issubset(cols), f"missing columns: {expected - cols}"
    store.close()


def test_migrations_are_idempotent_on_legacy_db(tmp_db: Path) -> None:
    """Simulate a pre-existing DB with the OLD schema, then upgrade."""
    legacy_schema = """
    CREATE TABLE anomaly_events (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id             TEXT    NOT NULL,
        event_type             TEXT    NOT NULL,
        ts_utc                 TEXT    NOT NULL,
        record_offset_seconds  REAL,
        residual               REAL    NOT NULL,
        threshold              REAL    NOT NULL,
        threshold_mode         TEXT    NOT NULL,
        model_version          TEXT    NOT NULL,
        acknowledged           INTEGER DEFAULT 0,
        ack_ts_utc             TEXT
    );
    """
    con = sqlite3.connect(tmp_db)
    con.executescript(legacy_schema)
    con.execute(
        "INSERT INTO anomaly_events(patient_id, event_type, ts_utc,"
        " residual, threshold, threshold_mode, model_version)"
        " VALUES ('100', 'anomaly_start', '2026-05-02T10:00:00Z',"
        " 0.5, 0.04, 'percentile', 'legacy')"
    )
    con.commit(); con.close()

    # First open: should run migrations and succeed.
    store = EventStore(tmp_db)
    cur = store._con.execute("PRAGMA table_info(anomaly_events)")
    cols = {r[1] for r in cur.fetchall()}
    assert "waveform" in cols
    assert "clinician_label" in cols
    # Legacy row is preserved
    rows = store.list_anomaly_starts(patient_id="100")
    assert len(rows) == 1
    assert rows[0].waveform is None  # legacy row had none
    store.close()

    # Second open: idempotent — must not raise on duplicate ALTERs.
    store2 = EventStore(tmp_db)
    store2.close()


def test_waveform_and_peak_round_trip(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    ev = _make_event()
    store.log_anomaly(ev)

    rows = store.list_anomaly_starts()
    assert len(rows) == 1
    got: StoredAnomaly = rows[0]
    assert got.patient_id == "100"
    assert got.peak_sample_index == ev.peak_sample_index
    assert got.waveform is not None
    np.testing.assert_array_equal(got.waveform, ev.waveform)
    # Reconstruction round-trips too (used for the orange overlay)
    assert got.reconstruction is not None
    np.testing.assert_array_equal(got.reconstruction, ev.reconstruction)
    store.close()


def test_clinician_decision_update(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event())
    [row] = store.list_anomaly_starts()
    store.set_clinician_decision(
        row.id, label="TP", notes="clear PVC, agreed",
        reviewed_ts_utc="2026-05-02T11:00:00Z",
        reviewer="Dr. Mészáros",
    )
    refetched = store.get_anomaly(row.id)
    assert refetched is not None
    assert refetched.clinician_label == "TP"
    assert refetched.clinician_notes == "clear PVC, agreed"
    assert refetched.reviewed_ts_utc == "2026-05-02T11:00:00Z"
    assert refetched.reviewer == "Dr. Mészáros"
    store.close()


def test_update_notes_does_not_change_decision(tmp_db: Path) -> None:
    """update_notes() saves a comment without committing TP/FP/Unsure."""
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event())
    [row] = store.list_anomaly_starts()
    store.update_notes(row.id, notes="hard to read, will revisit",
                       reviewer="Dr. Mészáros")
    refetched = store.get_anomaly(row.id)
    assert refetched is not None
    assert refetched.clinician_label is None       # decision untouched
    assert refetched.reviewed_ts_utc is None       # not yet reviewed
    assert refetched.clinician_notes == "hard to read, will revisit"
    assert refetched.reviewer == "Dr. Mészáros"
    store.close()


def test_update_notes_preserves_existing_reviewer_when_none_given(tmp_db: Path) -> None:
    """If update_notes is called without a reviewer, the existing one stays."""
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event())
    [row] = store.list_anomaly_starts()
    store.update_notes(row.id, notes="first pass", reviewer="Dr. M")
    store.update_notes(row.id, notes="second pass", reviewer=None)
    refetched = store.get_anomaly(row.id)
    assert refetched is not None
    assert refetched.clinician_notes == "second pass"
    assert refetched.reviewer == "Dr. M"  # preserved
    store.close()


def test_clinician_decision_rejects_invalid_labels(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event())
    [row] = store.list_anomaly_starts()
    with pytest.raises(ValueError):
        store.set_clinician_decision(row.id, label="MAYBE")
    store.close()


def test_legacy_event_without_waveform_still_inserts(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event(with_waveform=False))
    [row] = store.list_anomaly_starts()
    assert row.waveform is None
    assert row.peak_sample_index is None
    store.close()


def test_list_filters_by_patient(tmp_db: Path) -> None:
    store = EventStore(tmp_db)
    store.log_anomaly(_make_event(patient_id="100"))
    store.log_anomaly(_make_event(patient_id="105"))
    only_100 = store.list_anomaly_starts(patient_id="100")
    assert len(only_100) == 1 and only_100[0].patient_id == "100"
    only_105 = store.list_anomaly_starts(patient_id="105")
    assert len(only_105) == 1 and only_105[0].patient_id == "105"
    store.close()
