"""Render a doctor-friendly PDF that shows each flagged anomaly from the
events database in clinical context, so the cardiologist can audit
every alert herself.

For each anomaly_start row in logs/events.db:
- 6-second context window of the ECG (3 s before + 3 s after the flag)
- Raw mV signal on standard pink ECG-paper grid (1 mm = 0.04 s x 0.1 mV)
- The flag's beat-centre window highlighted in pale red
- Every annotated beat in view marked with its MIT-BIH symbol
- A textual panel listing residual, threshold, alert timestamp,
  the centre beat's symbol, and a YES / NO / NOT SURE checkbox the
  cardiologist can tick to audit the call

Usage:
    PYTHONPATH=. python tools/review_flagged_anomalies.py
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dataset import NORMAL_SYMBOLS


SYMBOL_NAMES = {
    "N": "Normal", "L": "LBBB", "R": "RBBB",
    "A": "Atrial premature", "a": "Aberrated atrial",
    "J": "Nodal premature", "S": "Supraventricular",
    "V": "PVC", "F": "Fusion (V+N)",
    "!": "V flutter", "e": "Atrial escape", "j": "Nodal escape",
    "E": "V escape", "/": "Paced", "f": "Fusion (P+N)",
    "Q": "Unclassifiable", "+": "Rhythm change",
    "~": "Signal quality", "|": "Isolated QRS-like",
}


def render_review(rec_id, signal, fs, ann, evt, page_num, total, pdf,
                  context_seconds=6.0, model_window_seconds=2.0):
    t_evt = float(evt["record_offset_seconds"])
    half_ctx = context_seconds / 2
    t0 = max(0.0, t_evt - half_ctx)
    t1 = min(len(signal) / fs, t_evt + half_ctx)
    s0 = int(t0 * fs)
    s1 = int(t1 * fs)
    seg = signal[s0:s1]
    t_axis = np.arange(s0, s1) / fs

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top: ECG with clinical grid
    ax = axes[0]
    ax.plot(t_axis, seg, color="#000000", lw=1.4)
    ax.set_xlim(t0, t1)
    mv_min, mv_max = float(seg.min()), float(seg.max())
    pad = 0.5 * (mv_max - mv_min) if mv_max > mv_min else 0.5
    ax.set_ylim(mv_min - pad * 0.2, mv_max + pad * 0.2)
    # Pink ECG grid
    for x in np.arange(t0, t1 + 1e-6, 0.04):
        ax.axvline(x, color="#f7c4c4", lw=0.4, zorder=0)
    for x in np.arange(t0, t1 + 1e-6, 0.20):
        ax.axvline(x, color="#e07b7b", lw=0.8, zorder=0)
    y0 = np.floor(ax.get_ylim()[0] * 10) / 10
    y1 = np.ceil(ax.get_ylim()[1] * 10) / 10
    for y in np.arange(y0, y1 + 1e-6, 0.1):
        ax.axhline(y, color="#f7c4c4", lw=0.4, zorder=0)
    for y in np.arange(y0, y1 + 1e-6, 0.5):
        ax.axhline(y, color="#e07b7b", lw=0.8, zorder=0)

    # Highlight the flag's 2-s model window in pale red
    half_model = model_window_seconds / 2
    ax.axvspan(t_evt - half_model, t_evt + half_model,
               color="#ffcccc", alpha=0.55, zorder=1,
               label="Model's 2-s alert window")

    # Mark every annotated beat in view
    ann_samples = np.asarray(ann.sample)
    in_view = (ann_samples >= s0) & (ann_samples < s1)
    centre_sym = "?"
    centre_dist = float("inf")
    for s, sym in zip(ann_samples[in_view], np.asarray(ann.symbol)[in_view]):
        t_pos = s / fs
        color = "magenta" if sym not in NORMAL_SYMBOLS else "green"
        ax.axvline(t_pos, ymin=0, ymax=0.06, color=color, lw=2, zorder=2)
        ax.text(t_pos, ax.get_ylim()[0] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                sym, ha="center", va="bottom",
                color=color, fontsize=11, fontweight="bold", zorder=3)
        d = abs(t_pos - t_evt)
        if d < centre_dist:
            centre_dist, centre_sym = d, sym

    ax.set_xlabel("time within recording (s)")
    ax.set_ylabel("amplitude (mV)")
    ax.set_title(
        f"Alert #{page_num}/{total} — record {rec_id} at t = "
        f"{int(t_evt)//60:02d}:{int(t_evt)%60:02d}.{int((t_evt%1)*100):02d}",
        fontsize=13, fontweight="bold",
    )

    # Bottom: text panel
    ax = axes[1]; ax.axis("off")
    info = [
        f"Record           {rec_id}    (MIT-BIH ID, NOT a patient identifier)",
        f"Time in record   {t_evt:.2f} s   (sample {int(t_evt*fs)})",
        f"Model residual   {evt['residual']:.4f}",
        f"Alert threshold  {evt['threshold']:.4f}",
        f"Mode             {evt['threshold_mode']}",
        f"Centre beat      {centre_sym} - {SYMBOL_NAMES.get(centre_sym, '?')}",
        f"Wall-clock       {evt['ts_utc']}",
        "",
        "Cardiologist audit:    [ ] CORRECT FLAG    [ ] FALSE ALARM    [ ] NOT SURE",
        "Notes: ____________________________________________________________________",
    ]
    ax.text(0.02, 0.95, "\n".join(info), va="top", ha="left",
            fontfamily="monospace", fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="logs/events.db")
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--out", default="docs/clinical_proof/flagged_review.pdf")
    p.add_argument("--record", default=None,
                   help="filter to one record id")
    p.add_argument("--max-events", type=int, default=20)
    p.add_argument("--context-seconds", type=float, default=6.0)
    args = p.parse_args()

    if not Path(args.db).exists():
        print(f"No DB at {args.db}. Run the pipeline first."); return 1

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    sql = "SELECT * FROM anomaly_events WHERE event_type='anomaly_start'"
    params: list = []
    if args.record:
        sql += " AND patient_id=?"; params.append(args.record)
    sql += " ORDER BY id"

    all_evts = con.execute(sql, params).fetchall()

    # Deduplicate alerts that map to the same physical event in the
    # recording (same record + same record_offset_seconds rounded to ms)
    seen = set(); evts = []
    for e in all_evts:
        key = (e["patient_id"], round(e["record_offset_seconds"] or -1, 3))
        if key in seen: continue
        seen.add(key); evts.append(e)
    evts = evts[:args.max_events]

    if not evts:
        print("No anomaly_start events to review."); return 0

    # Group by record id and load each record once
    records: dict = {}
    for rid in {e["patient_id"] for e in evts}:
        rec_path = str(Path(args.data_dir) / rid)
        if not Path(f"{rec_path}.dat").exists():
            print(f"  skip record {rid}: file missing"); continue
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")
        records[rid] = (rec.p_signal[:, 0].astype(float), int(rec.fs), ann)

    sym_counter: Counter = Counter()
    by_record: Counter = Counter()
    for e in evts:
        if e["patient_id"] not in records: continue
        sig, fs, ann = records[e["patient_id"]]
        t = float(e["record_offset_seconds"])
        ann_samples = np.asarray(ann.sample)
        nearest = int(np.argmin(np.abs(ann_samples - t * fs)))
        sym = ann.symbol[nearest]
        sym_counter[sym] += 1
        by_record[e["patient_id"]] += 1

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(evts)} alerts to {out_path}...")

    with PdfPages(out_path) as pdf:
        # Cover
        cover = plt.figure(figsize=(11, 7))
        cover.text(0.5, 0.93,
                   "Audit of model-flagged anomalies",
                   ha="center", fontsize=18, fontweight="bold")
        text = (
            f"Each page shows one alert the model raised, in clinical context\n"
            f"(6-second window, raw mV on standard ECG paper grid).\n\n"
            f"Total alerts in this PDF: {len(evts)}\n"
            f"Records covered: " + ", ".join(sorted(by_record)) + "\n\n"
            "Centre beats hit by these alerts (cardiologist annotations):\n"
        )
        for sym, n in sym_counter.most_common():
            text += f"  {sym} {SYMBOL_NAMES.get(sym, '?'):26}  {n}\n"
        text += (
            "\nLegend on each page:\n"
            "  Black trace        the ECG (raw mV)\n"
            "  Pink shaded box    the 2-second window the model decided on\n"
            "  Green tick         annotated NORMAL beat (cardiologist label)\n"
            "  Magenta tick       annotated ABNORMAL beat (cardiologist label)\n"
            "  Letter under tick  MIT-BIH beat symbol\n\n"
            "How to use this PDF:\n"
            "  1. Look at the pink window: that is what the model flagged.\n"
            "  2. Look at the magenta/green ticks INSIDE the pink window.\n"
            "  3. If there is a magenta tick inside or right at the edge,\n"
            "     the model probably caught a real abnormality.\n"
            "  4. If everything inside the pink window is green, the alert\n"
            "     is likely a false alarm.\n"
            "  5. Tick CORRECT FLAG / FALSE ALARM / NOT SURE on each page.\n"
        )
        cover.text(0.06, 0.85, text, va="top", fontfamily="monospace",
                   fontsize=10)
        pdf.savefig(cover); plt.close(cover)

        for i, e in enumerate(evts, start=1):
            if e["patient_id"] not in records: continue
            sig, fs, ann = records[e["patient_id"]]
            render_review(e["patient_id"], sig, fs, ann, e, i, len(evts),
                          pdf, context_seconds=args.context_seconds)

    print(f"Saved {out_path}")
    print(f"  Records: {dict(by_record)}")
    print(f"  Hit symbols: {dict(sym_counter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
