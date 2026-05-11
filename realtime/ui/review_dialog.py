"""Cardiologist review dialog for past flagged anomalies.

Mirrors the layout of the offline ``tools/review_flagged_anomalies.py``
PDF but inside the live PyQt application: the cardiologist clicks the
"Review flagged anomalies" button on the main window, sees a list of
past alerts (newest first), selects one, sees TWO synchronised views —
a 10-second mV clinical-context view (with ECG-paper grid, time axis
in mm:ss.cc, the model's 2-s alert window highlighted) and the model's
own 2-s view with the autoencoder reconstruction overlaid in orange —
and adjudicates the alert as TP / FP / Unsure.

The clinician's decision is written back into the same SQLite
``anomaly_events`` row via ``EventStore.set_clinician_decision`` so
the kappa-replication workflow no longer requires the offline PDF.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from realtime.event_store import EventStore, StoredAnomaly

log = logging.getLogger(__name__)
MITBIH_DIR_DEFAULT = Path("data/mitbih")
CONTEXT_SECONDS = 10.0
MODEL_WINDOW_SECONDS = 2.0


_QSETTINGS_ORG = "ECGAEProject"
_QSETTINGS_APP = "RealtimeMonitor"
_QSETTINGS_REVIEWER_KEY = "review/last_reviewer"


WINDOW_SAMPLES = 720           # 2 s @ 360 Hz, matches training
SAMPLING_RATE  = 360


class ReviewDialog(QDialog):
    """Modal dialog for reviewing past anomaly_start events."""

    def __init__(self, store: EventStore, parent=None) -> None:
        super().__init__(parent)
        self._store = store
        self._items: List[StoredAnomaly] = []
        self._current: Optional[StoredAnomaly] = None
        self.setWindowTitle("Review flagged anomalies")
        # Wider default so the matplotlib y-axis tick labels (mV / σ)
        # fit without crowding the plot area.
        self.resize(1400, 760)
        self._build_ui()
        self._reload()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Top bar: reviewer ID, patient filter, count, refresh.
        top = QHBoxLayout()
        top.addWidget(QLabel("Reviewer (name / ID):"))
        self._reviewer_input = QLineEdit()
        self._reviewer_input.setPlaceholderText(
            "e.g. Dr. Mészáros Henrietta — saved with each decision"
        )
        self._reviewer_input.setMinimumWidth(280)
        # Persist the reviewer across sessions via QSettings so the
        # cardiologist does not retype it every time the dialog opens.
        settings = QSettings(_QSETTINGS_ORG, _QSETTINGS_APP)
        last = settings.value(_QSETTINGS_REVIEWER_KEY, "", type=str)
        if last:
            self._reviewer_input.setText(last)
        self._reviewer_input.editingFinished.connect(self._persist_reviewer)
        top.addWidget(self._reviewer_input)

        top.addSpacing(20)
        top.addWidget(QLabel("Record ID:"))
        self._patient_combo = QComboBox()
        self._patient_combo.addItem("All records", None)
        self._patient_combo.setToolTip(
            "MIT-BIH record identifier (e.g. 100, 208, 222). NOT a hospital "
            "patient ID — no real patient identity is stored in this dataset."
        )
        self._patient_combo.currentIndexChanged.connect(self._reload)
        top.addWidget(self._patient_combo)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #666;")
        top.addSpacing(12); top.addWidget(self._count_label)
        top.addStretch()
        reload_btn = QPushButton("Refresh")
        reload_btn.clicked.connect(self._reload)
        top.addWidget(reload_btn)
        root.addLayout(top)

        # Main split: list on the left, plot + decision on the right
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, stretch=1)

        # --- left: list of anomalies ---
        left = QWidget(); left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Flagged anomalies (newest first):"))
        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._on_selection)
        left_l.addWidget(self._list)
        split.addWidget(left)

        # --- right: plot + metadata + decision row ---
        right = QWidget(); right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel("Select an alert from the list →")
        self._title.setStyleSheet("font-weight: bold; font-size: 13pt;")
        right_l.addWidget(self._title)

        # Matplotlib canvas with two stacked axes:
        #   ax_context: 10-s mV view loaded from data/mitbih/<rec>.dat with
        #               pink ECG-paper grid, alert window highlighted.
        #               (mirrors tools/review_flagged_anomalies.py)
        #   ax_model:   the 2-s window the model actually saw, with the
        #               input (blue) and the reconstruction (orange) overlaid
        #               and the per-sample peak marked with a dashed red line.
        # constrained_layout keeps tick labels from being clipped when the
        # dialog is resized; figsize is generous so y-axis labels fit.
        self._fig = Figure(figsize=(11, 7), constrained_layout=True)
        self._ax_context, self._ax_model = self._fig.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 2]},
        )
        self._canvas = FigureCanvas(self._fig)
        # Give the canvas a comfortable minimum so it doesn't get squeezed
        # by the metadata block underneath.
        self._canvas.setMinimumHeight(380)
        right_l.addWidget(self._canvas, stretch=1)

        self._meta = QLabel("")
        self._meta.setStyleSheet("color: #333; font-family: monospace;")
        self._meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_l.addWidget(self._meta)

        # Decision row
        decision_row = QHBoxLayout()
        decision_row.addWidget(QLabel("Cardiologist decision:"))
        self._tp_btn = QPushButton("✓ True positive")
        self._fp_btn = QPushButton("✗ False alarm")
        self._un_btn = QPushButton("? Unsure / unreadable")
        self._tp_btn.clicked.connect(lambda: self._save_decision("TP"))
        self._fp_btn.clicked.connect(lambda: self._save_decision("FP"))
        self._un_btn.clicked.connect(lambda: self._save_decision("UNSURE"))
        for b in (self._tp_btn, self._fp_btn, self._un_btn):
            b.setEnabled(False)
            decision_row.addWidget(b)
        decision_row.addStretch()
        right_l.addLayout(decision_row)

        # Notes row + dedicated "Save notes only" button so the cardiologist
        # can persist a comment without committing to TP/FP/Unsure.
        notes_header = QHBoxLayout()
        notes_header.addWidget(QLabel("Notes:"))
        notes_header.addStretch()
        self._save_notes_btn = QPushButton("💾 Save notes")
        self._save_notes_btn.setToolTip(
            "Save just the notes for this alert without changing the decision"
        )
        self._save_notes_btn.setEnabled(False)
        self._save_notes_btn.clicked.connect(self._save_notes_only)
        notes_header.addWidget(self._save_notes_btn)
        right_l.addLayout(notes_header)

        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(70)
        self._notes.setPlaceholderText(
            "Free-text notes on this alert. Saved with the decision when you "
            "click TP/FP/Unsure, or independently with the 💾 Save notes button."
        )
        right_l.addWidget(self._notes)

        split.addWidget(right)
        # Give the right pane (plot + metadata + decision) the bulk of the
        # horizontal real estate so the matplotlib y-axis labels fit.
        split.setSizes([280, 1120])

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        patient = self._patient_combo.currentData()
        self._items = self._store.list_anomaly_starts(
            patient_id=patient, limit=500,
        )
        self._populate_patient_combo()
        self._list.clear()
        for ev in self._items:
            self._list.addItem(_format_list_row(ev))
        n_total = len(self._items)
        n_review = sum(1 for e in self._items if e.clinician_label)
        self._count_label.setText(f"{n_total} alerts — {n_review} already reviewed")
        # Clear preview
        self._current = None
        self._ax_context.clear(); self._ax_model.clear()
        self._ax_context.set_xticks([]); self._ax_context.set_yticks([])
        self._ax_model.set_xticks([]); self._ax_model.set_yticks([])
        self._canvas.draw_idle()
        self._title.setText("Select an alert from the list →")
        self._meta.setText("")
        for b in (self._tp_btn, self._fp_btn, self._un_btn,
                  self._save_notes_btn):
            b.setEnabled(False)
        self._notes.clear()

    def _populate_patient_combo(self) -> None:
        # Build unique patient list from the unfiltered DB snapshot so the
        # combo reflects every patient with alerts, not only the current
        # filter result.
        all_items = self._store.list_anomaly_starts(patient_id=None, limit=2000)
        patients = sorted({e.patient_id for e in all_items})
        current = self._patient_combo.currentData()
        self._patient_combo.blockSignals(True)
        self._patient_combo.clear()
        self._patient_combo.addItem("All patients", None)
        for p in patients:
            self._patient_combo.addItem(p, p)
        # Restore selection
        if current is not None:
            idx = self._patient_combo.findData(current)
            if idx >= 0:
                self._patient_combo.setCurrentIndex(idx)
        self._patient_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------

    def _on_selection(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._items):
            return
        self._current = self._items[row]
        self._render_current()

    def _render_current(self) -> None:
        ev = self._current
        if ev is None:
            return
        self._title.setText(
            f"Alert #{ev.id} — Record ID {ev.patient_id}  ·  "
            f"residual {ev.residual:.4f}  >  threshold {ev.threshold:.4f}"
        )
        self._draw_context_view(ev)
        self._draw_model_view(ev)
        self._canvas.draw_idle()

        self._meta.setText(_format_metadata(ev))
        self._notes.setPlainText(ev.clinician_notes or "")
        for b in (self._tp_btn, self._fp_btn, self._un_btn, self._save_notes_btn):
            b.setEnabled(True)

    def _draw_context_view(self, ev: StoredAnomaly) -> None:
        """Top axis: 10-s mV context loaded from data/mitbih/<rec>.dat
        with pink ECG-paper grid and the model's 2-s alert window
        highlighted. Falls back to a centred message when the source
        recording cannot be loaded."""
        ax = self._ax_context
        ax.clear()
        if ev.record_offset_seconds is None:
            ax.text(0.5, 0.5,
                    "No record offset stored — cannot render clinical context.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            return
        try:
            seg_mv, t0, t1, fs = _load_mv_context(
                ev.patient_id, float(ev.record_offset_seconds),
                half_window=CONTEXT_SECONDS / 2,
            )
        except Exception as e:
            log.info("could not load mV context for %s: %s", ev.patient_id, e)
            ax.text(0.5, 0.5,
                    f"Source recording not available "
                    f"(data/mitbih/{ev.patient_id}.dat).\n"
                    f"Clinical-context view requires the original WFDB files.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            return

        # X-axis is RELATIVE to the start of the saved 10-s context (0..10 s).
        # The alert sits at the centre, around t_rel = 5 s — clinically the
        # natural "where in the strip is the abnormality" question.
        rel_axis = np.arange(len(seg_mv)) / fs
        rel_end = float(rel_axis[-1]) if len(rel_axis) else (t1 - t0)
        ax.plot(rel_axis, seg_mv, color="#000000", lw=1.4, zorder=2)

        mv_min, mv_max = float(seg_mv.min()), float(seg_mv.max())
        pad = max(0.5 * (mv_max - mv_min), 0.5)
        ax.set_ylim(mv_min - pad * 0.2, mv_max + pad * 0.2)
        ax.set_xlim(0.0, rel_end)

        # Pink ECG-paper grid: minor every 0.04 s / 0.1 mV, major every 0.2 s / 0.5 mV
        for x in np.arange(0.0, rel_end + 1e-9, 0.04):
            ax.axvline(x, color="#f7c4c4", lw=0.4, zorder=0)
        for x in np.arange(0.0, rel_end + 1e-9, 0.20):
            ax.axvline(x, color="#e07b7b", lw=0.8, zorder=0)
        y0 = np.floor(ax.get_ylim()[0] * 10) / 10
        y1 = np.ceil(ax.get_ylim()[1] * 10) / 10
        for y in np.arange(y0, y1 + 1e-9, 0.1):
            ax.axhline(y, color="#f7c4c4", lw=0.4, zorder=0)
        for y in np.arange(y0, y1 + 1e-9, 0.5):
            ax.axhline(y, color="#e07b7b", lw=0.8, zorder=0)

        # Highlight the model's 2-s alert window. The alert is centred on
        # ``record_offset_seconds`` and the saved context is centred on the
        # same value, so on the relative axis the alert sits around 5 s.
        t_evt = float(ev.record_offset_seconds)
        alert_centre_rel = t_evt - t0
        half_model = MODEL_WINDOW_SECONDS / 2
        ax.axvspan(alert_centre_rel - half_model,
                   alert_centre_rel + half_model,
                   color="#ffcccc", alpha=0.55, zorder=1,
                   label="model's 2-s alert window")

        # Lock major/minor ticks to ECG-paper convention so the labelled
        # numbers on the y-axis read 0.0, 0.5, 1.0, 1.5 ... mV instead of
        # matplotlib's default 1 mV cadence; minor ticks at 0.1 mV.
        ax.xaxis.set_major_locator(MultipleLocator(1.0))   # 1 s major label
        ax.xaxis.set_minor_locator(MultipleLocator(0.20))  # 0.2 s minor (matches grid)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))   # 0.5 mV major
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))   # 0.1 mV minor

        ax.set_xlabel("time within saved 10-s window (s)  —  alert at t ≈ "
                      f"{alert_centre_rel:.1f} s", fontsize=9)
        ax.set_ylabel("amplitude (mV)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="both", which="minor", length=2)

        # Title carries the absolute context: the MIT-BIH record ID, the
        # offset within that recording, and the wall-clock UTC date/time the
        # alert fired (so the cardiologist can correlate with the shift log).
        wall = _format_wallclock(ev.ts_utc)
        ax.set_title(
            f"10 s clinical context — Record ID {ev.patient_id} (MIT-BIH, "
            f"not a hospital patient ID)  ·  alert at "
            f"{_fmt_offset(t_evt)} of recording  ·  {wall}",
            fontsize=9, fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    def _draw_model_view(self, ev: StoredAnomaly) -> None:
        """Bottom axis: the exact 2-s window the autoencoder saw, with its
        reconstruction overlaid (orange) and the per-sample peak error
        marked with a vertical dashed red line. Y-axis is z-scored σ
        (the model's input space) — that is what was actually compared
        to produce the residual, so this is the honest model view."""
        ax = self._ax_model
        ax.clear()
        if ev.waveform is None or len(ev.waveform) == 0:
            ax.text(0.5, 0.5, "No saved waveform for this alert.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            return
        n = len(ev.waveform)
        t_local = np.arange(n) / float(SAMPLING_RATE)  # seconds within window
        ax.plot(t_local, ev.waveform, color="#1f77b4", lw=1.6, label="input")
        if ev.reconstruction is not None and len(ev.reconstruction) == n:
            ax.plot(t_local, ev.reconstruction,
                    color="#ff7f0e", lw=1.4, linestyle="--",
                    label="autoencoder reconstruction")
        if ev.peak_sample_index is not None and 0 <= ev.peak_sample_index < n:
            ax.axvline(t_local[ev.peak_sample_index],
                       color="#d62728", lw=1.5, linestyle=":",
                       label=f"peak error sample "
                             f"({ev.peak_sample_index/SAMPLING_RATE*1000:.0f} ms)")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("time within model window (ms — 0 to 2000)", fontsize=9)
        ax.set_ylabel("amplitude (σ, z-scored)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(
            "Model's 2-s view — what the autoencoder actually saw "
            "(blue = input, orange dashed = reconstruction, red dotted = peak)",
            fontsize=9, fontweight="bold",
        )
        # Convert tick labels to milliseconds for the cardiologist's eye.
        xticks = np.arange(0, t_local[-1] + 1e-6, 0.2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(round(t*1000))}" for t in xticks])
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # ------------------------------------------------------------------
    # Decision persistence
    # ------------------------------------------------------------------

    def _save_decision(self, label: str) -> None:
        if self._current is None:
            return
        reviewer = self._current_reviewer()
        if not reviewer:
            QMessageBox.warning(
                self, "Reviewer required",
                "Enter your name or ID in the 'Reviewer' field at the top "
                "of this dialog before recording a decision.",
            )
            self._reviewer_input.setFocus()
            return
        ts = datetime.now(tz=timezone.utc).isoformat()
        try:
            self._store.set_clinician_decision(
                self._current.id, label=label,
                notes=self._notes.toPlainText() or None,
                reviewed_ts_utc=ts,
                reviewer=reviewer,
            )
        except Exception as e:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Could not save decision", str(e))
            return
        self._persist_reviewer()
        eid = self._current.id
        self._reload()
        for i, ev in enumerate(self._items):
            if ev.id == eid:
                self._list.setCurrentRow(i)
                break

    def _save_notes_only(self) -> None:
        """Persist the notes for the currently-selected alert WITHOUT
        changing its TP/FP/UNSURE decision."""
        if self._current is None:
            return
        try:
            self._store.update_notes(
                self._current.id,
                notes=self._notes.toPlainText() or None,
                reviewer=self._current_reviewer() or None,
            )
        except Exception as e:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Could not save notes", str(e))
            return
        self._persist_reviewer()
        # Refresh the in-memory view so subsequent re-selections show the
        # newly-saved notes.
        eid = self._current.id
        self._reload()
        for i, ev in enumerate(self._items):
            if ev.id == eid:
                self._list.setCurrentRow(i)
                break
        # Brief confirmation in the title bar.
        self.setWindowTitle("Review flagged anomalies — notes saved")
        # Flick title back after a short delay; cheap and avoids a popup.
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(
            1500, lambda: self.setWindowTitle("Review flagged anomalies"),
        )

    def _current_reviewer(self) -> str:
        return self._reviewer_input.text().strip()

    def _persist_reviewer(self) -> None:
        QSettings(_QSETTINGS_ORG, _QSETTINGS_APP).setValue(
            _QSETTINGS_REVIEWER_KEY, self._current_reviewer(),
        )


# ----------------------------------------------------------------------
# Free-function formatters (kept module-level so they can be unit-tested)
# ----------------------------------------------------------------------

def _format_list_row(ev: StoredAnomaly) -> QListWidgetItem:
    badge = (
        "✓ TP" if ev.clinician_label == "TP"
        else "✗ FP" if ev.clinician_label == "FP"
        else "? UN" if ev.clinician_label == "UNSURE"
        else "·    "
    )
    when = ev.ts_utc.split("T")[-1][:8] if "T" in ev.ts_utc else ev.ts_utc
    text = (
        f"{badge}  Record ID {ev.patient_id}   "
        f"t={_fmt_offset(ev.record_offset_seconds)}   "
        f"resid={ev.residual:.3f}   {when}"
    )
    item = QListWidgetItem(text)
    if ev.clinician_label == "TP":
        item.setForeground(QColor("#1a7a1a"))
    elif ev.clinician_label == "FP":
        item.setForeground(QColor("#a02020"))
    elif ev.clinician_label == "UNSURE":
        item.setForeground(QColor("#7a6020"))
    return item


def _format_metadata(ev: StoredAnomaly) -> str:
    lines = [
        f"Alert ID         {ev.id}",
        f"Record ID        {ev.patient_id}    (MIT-BIH dataset record — NOT a hospital patient ID)",
        f"Patient ID       n/a    (no real patient identity is stored in this research dataset)",
        f"Time in record   {_fmt_offset(ev.record_offset_seconds)}",
        f"Wall-clock UTC   {_format_wallclock(ev.ts_utc)}    (raw: {ev.ts_utc})",
        f"Residual         {ev.residual:.6f}",
        f"Threshold        {ev.threshold:.6f}   (mode: {ev.threshold_mode})",
        f"Peak sample idx  {ev.peak_sample_index if ev.peak_sample_index is not None else 'n/a'}"
        + (f"   (≈ {ev.peak_sample_index/SAMPLING_RATE:.3f} s into the window)"
           if ev.peak_sample_index is not None else ""),
        f"Model            {ev.model_version}",
    ]
    if ev.clinician_label:
        who = ev.reviewer or "(reviewer not recorded)"
        lines.append(
            f"Previous review  {ev.clinician_label} at {ev.reviewed_ts_utc}"
            f"  by {who}"
        )
    elif ev.reviewer:
        # Notes-only save without a TP/FP/Unsure decision.
        lines.append(f"Last edit by     {ev.reviewer}")
    return "\n".join(lines)


def _fmt_offset(off: Optional[float]) -> str:
    if off is None:
        return "n/a"
    s = max(0, int(off))
    return f"{s // 60:02d}:{s % 60:02d}.{int((off - s) * 100):02d}"


def _format_wallclock(ts_utc: Optional[str]) -> str:
    """Render a stored ISO-8601 wall-clock timestamp as ``YYYY-MM-DD HH:MM UTC``.

    Returns the original string verbatim if it cannot be parsed (e.g. legacy
    rows where the timestamp was written in a non-ISO format).
    """
    if not ts_utc:
        return "n/a"
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return ts_utc


def _load_mv_context(
    record_id: str,
    t_center: float,
    *,
    half_window: float = CONTEXT_SECONDS / 2,
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Load a slice of the original mV ECG signal from the local WFDB
    record so the cardiologist can see what was happening *around* the
    alert, not just inside the model's 2-s window.

    Returns ``(seg_mv, t_start_s, t_end_s, fs_hz)`` — the raw mV samples
    of the first lead (MLII), the wall-clock-relative start and end
    seconds, and the sampling rate. Raises if the .dat/.hea pair is not
    on disk.
    """
    import wfdb  # imported lazily so unit tests that don't load mV work
    base = Path(data_dir or MITBIH_DIR_DEFAULT) / record_id
    rec = wfdb.rdrecord(str(base))
    fs = float(rec.fs)
    signal_mv = rec.p_signal[:, 0].astype(float)
    t0 = max(0.0, t_center - half_window)
    t1 = min(len(signal_mv) / fs, t_center + half_window)
    s0, s1 = int(t0 * fs), int(t1 * fs)
    if s1 <= s0:
        raise ValueError(f"empty slice for {record_id} at t={t_center}")
    return signal_mv[s0:s1], t0, t1, fs
