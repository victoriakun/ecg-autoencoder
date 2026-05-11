"""Tests for the cardiologist review dialog.

Pure-function tests (formatters) run unconditionally. The Qt smoke test
runs only when PyQt5 + a usable display are available; it is skipped
otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from realtime.event_store import AnomalyEvent, EventStore, StoredAnomaly
from realtime.ui.review_dialog import (
    SAMPLING_RATE, _fmt_offset, _format_metadata, _format_wallclock,
)


# ----------------------------------------------------------------------
# Pure-function tests (no Qt needed)
# ----------------------------------------------------------------------

def test_fmt_offset_handles_none():
    assert _fmt_offset(None) == "n/a"


def test_fmt_offset_renders_minutes_and_seconds():
    assert _fmt_offset(0.0).startswith("00:00")
    s = _fmt_offset(125.42)
    assert s.startswith("02:05") and s.endswith(".42")


def _make_stored(**over) -> StoredAnomaly:
    base = dict(
        id=1, patient_id="100", event_type="anomaly_start",
        ts_utc="2026-05-02T10:00:00Z", record_offset_seconds=42.5,
        residual=0.123, threshold=0.043, threshold_mode="percentile",
        model_version="test@sha=abc",
        waveform=np.zeros(720, dtype=np.float32),
        reconstruction=np.zeros(720, dtype=np.float32),
        peak_sample_index=360, clinician_label=None, clinician_notes=None,
        reviewed_ts_utc=None, reviewer=None,
    )
    base.update(over)
    return StoredAnomaly(**base)


def test_metadata_formatter_includes_all_fields():
    s = _format_metadata(_make_stored())
    assert "Alert ID" in s and "100" in s and "0.123000" in s
    assert "Peak sample idx  360" in s
    assert f"≈ {360/SAMPLING_RATE:.3f} s" in s
    # Mode is shown alongside threshold
    assert "percentile" in s


def test_metadata_distinguishes_record_id_from_patient_id():
    """The MIT-BIH record ID is *not* a hospital patient ID; must not conflate."""
    s = _format_metadata(_make_stored())
    assert "Record ID        100" in s
    # The legacy "Patient (record)" wording is GONE.
    assert "Patient (record)" not in s
    # And there is an explicit Patient ID row showing it is not stored.
    assert "Patient ID       n/a" in s
    # Plus a clarifying note about MIT-BIH being a research dataset.
    assert "NOT a hospital patient ID" in s


def test_format_wallclock_formats_iso_to_humane():
    out = _format_wallclock("2026-05-02T18:43:07Z")
    assert out == "2026-05-02 18:43 UTC"


def test_format_wallclock_passes_through_unparseable():
    raw = "garbage-not-a-date"
    assert _format_wallclock(raw) == raw


def test_format_wallclock_handles_none():
    assert _format_wallclock(None) == "n/a"


def test_metadata_handles_missing_peak():
    s = _format_metadata(_make_stored(peak_sample_index=None))
    assert "Peak sample idx  n/a" in s


def test_metadata_includes_prior_review_when_present():
    s = _format_metadata(_make_stored(
        clinician_label="TP", reviewed_ts_utc="2026-05-02T11:00:00Z",
        reviewer="Dr. Mészáros",
    ))
    assert "Previous review  TP at 2026-05-02T11:00:00Z" in s
    assert "by Dr. Mészáros" in s


def test_metadata_marks_missing_reviewer():
    s = _format_metadata(_make_stored(
        clinician_label="FP", reviewed_ts_utc="2026-05-02T11:00:00Z",
        reviewer=None,
    ))
    assert "(reviewer not recorded)" in s


def test_metadata_shows_notes_only_reviewer():
    """If only notes were saved (no decision yet), still display reviewer."""
    s = _format_metadata(_make_stored(
        clinician_label=None, reviewer="Dr. M",
    ))
    assert "Last edit by     Dr. M" in s


# ----------------------------------------------------------------------
# Qt smoke test (needs PyQt5 + a usable display or offscreen platform)
# ----------------------------------------------------------------------

def _qt_available() -> bool:
    try:
        import PyQt5  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _qt_available(), reason="PyQt5 not installed")
def test_review_dialog_smoke(tmp_path: Path) -> None:
    """End-to-end: log two anomalies, open the dialog, save a decision."""
    # Force the offscreen Qt platform so this works on headless CI.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    store = EventStore(tmp_path / "events.db")
    rng = np.random.default_rng(0)
    for i in range(2):
        wf = rng.standard_normal(720).astype(np.float32)
        peak = int(np.abs(wf).argmax())
        store.log_anomaly(AnomalyEvent(
            patient_id=f"10{i}", event_type="anomaly_start",
            ts_utc=f"2026-05-02T10:00:0{i}Z",
            residual=0.5 + 0.1 * i, threshold=0.04, threshold_mode="percentile",
            model_version="test", record_offset_seconds=10.0 * i,
            waveform=wf, peak_sample_index=peak,
        ))

    from realtime.ui.review_dialog import ReviewDialog
    dlg = ReviewDialog(store)

    # Two items should appear in the list (newest first).
    assert dlg._list.count() == 2
    # Select the first item; preview must populate.
    dlg._list.setCurrentRow(0)
    app.processEvents()
    assert dlg._current is not None
    assert dlg._tp_btn.isEnabled()
    assert dlg._save_notes_btn.isEnabled()
    # Both axes were drawn — title check is the cheapest sanity assertion.
    assert "Model's 2-s view" in dlg._ax_model.get_title()

    # Reviewer is required for a decision.
    dlg._reviewer_input.setText("Dr. Test")

    # Save a TP decision and confirm it round-trips back into the DB.
    dlg._notes.setPlainText("clear PVC, agreed")
    dlg._save_decision("TP")
    refreshed = store.get_anomaly(dlg._current.id) if dlg._current else None
    assert refreshed is not None
    assert refreshed.clinician_label == "TP"
    assert refreshed.clinician_notes == "clear PVC, agreed"
    assert refreshed.reviewer == "Dr. Test"

    # Notes-only save on the OTHER alert (no decision change).
    other_id = next(e.id for e in dlg._items if e.id != refreshed.id)
    for i, e in enumerate(dlg._items):
        if e.id == other_id:
            dlg._list.setCurrentRow(i); break
    app.processEvents()
    dlg._notes.setPlainText("hard to read, will revisit")
    dlg._save_notes_only()
    other = store.get_anomaly(other_id)
    assert other is not None
    assert other.clinician_label is None       # decision untouched
    assert other.clinician_notes == "hard to read, will revisit"
    assert other.reviewer == "Dr. Test"

    dlg.close()
    store.close()


@pytest.mark.skipif(not _qt_available(), reason="PyQt5 not installed")
def test_review_dialog_requires_reviewer_for_decision(tmp_path: Path) -> None:
    """Clicking TP/FP/Unsure with an empty reviewer field must NOT save."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    # Clear any previously-stored reviewer so the field starts blank.
    QSettings("ECGAEProject", "RealtimeMonitor").setValue(
        "review/last_reviewer", "")

    store = EventStore(tmp_path / "events.db")
    rng = np.random.default_rng(0)
    wf = rng.standard_normal(720).astype(np.float32)
    store.log_anomaly(AnomalyEvent(
        patient_id="100", event_type="anomaly_start",
        ts_utc="2026-05-02T10:00:00Z",
        residual=0.5, threshold=0.04, threshold_mode="percentile",
        model_version="t", record_offset_seconds=10.0,
        waveform=wf, peak_sample_index=int(np.abs(wf).argmax()),
    ))
    from realtime.ui.review_dialog import ReviewDialog
    dlg = ReviewDialog(store)
    dlg._reviewer_input.setText("")  # explicitly blank
    dlg._list.setCurrentRow(0)
    app.processEvents()
    # Patch QMessageBox.warning so the test doesn't pop a real dialog.
    from PyQt5.QtWidgets import QMessageBox
    calls = []
    QMessageBox.warning = staticmethod(lambda *a, **k: calls.append(a))
    dlg._save_decision("TP")
    # Decision must NOT have been saved.
    refreshed = store.get_anomaly(dlg._current.id)
    assert refreshed is not None
    assert refreshed.clinician_label is None
    assert len(calls) == 1, "expected the missing-reviewer warning to fire"
    dlg.close()
    store.close()
