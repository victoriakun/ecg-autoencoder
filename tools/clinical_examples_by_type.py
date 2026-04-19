"""Render two PDFs of ~50 example beats grouped by arrhythmia type.

For discussion with the cardiologist about which categories the model
catches vs misses. Uses the full Option-C runtime pipeline (XQRS beat
detection + warmup global normalize + threshold 0.0434).

Outputs:
  docs/clinical_proof/examples_by_type.pdf       (original ECG only)
  docs/clinical_proof/examples_by_type_recon.pdf (with reconstruction overlay)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wfdb
from wfdb.processing import XQRS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dataset import NORMAL_SYMBOLS, ANOMALY_SYMBOLS
from models import ConvAutoencoder
from preprocess import bandpass_filter
from realtime.inference import load_model
from realtime.normalizer import WarmupNormalizer


SYMBOL_NAMES = {
    "N": "Normal sinus beat",
    "L": "Left bundle branch block",
    "R": "Right bundle branch block",
    "A": "Atrial premature beat (APC)",
    "a": "Aberrated atrial premature",
    "J": "Nodal (junctional) premature",
    "S": "Supraventricular premature",
    "V": "Premature ventricular contraction (PVC)",
    "F": "Fusion of ventricular and normal",
    "!": "Ventricular flutter wave",
    "e": "Atrial escape",
    "j": "Nodal escape",
    "E": "Ventricular escape",
    "/": "Paced beat",
    "f": "Fusion of paced and normal",
    "Q": "Unclassifiable",
}

# Target distribution (~50 beats total)
TARGETS = {
    "N":   8,   # normals (mostly TN, a few FP for contrast)
    "V":   8,   # PVC - model usually catches
    "F":   8,   # fusion - model usually misses
    "A":   6,   # atrial premature - mixed
    "L":   8,   # LBBB - left bundle branch block
    "R":   5,   # RBBB - right bundle branch block
    "/":   4,   # paced
    "f":   2,   # fusion paced
}

# Records that contain the various symbols in MIT-BIH ARR
RECORDS = ["100", "109", "111", "118", "119", "200", "207", "208",
           "209", "212", "213", "214", "217", "222", "232"]


def collect_beats(data_dir: Path, model, threshold: float, warmup: int):
    """Run runtime pipeline on each record, collect
    (rec_id, p, residual, center_sym, pre, recon, raw_mv, fs, beats_in_window)
    where `beats_in_window` is a list of (offset_samples, symbol) for every
    annotated beat falling inside the 2-second window."""
    out: list = []
    for rid in RECORDS:
        rec_path = str(data_dir / rid)
        try:
            rec = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            print(f"  skip {rid}: {e}"); continue
        sig = rec.p_signal[:, 0].astype(float)
        fs = int(rec.fs)
        xqrs = XQRS(sig=sig, fs=fs); xqrs.detect(verbose=False)
        peaks = list(xqrs.qrs_inds)
        norm = WarmupNormalizer(warmup)
        WIN = 720; half = WIN // 2
        ann_samples = np.asarray(ann.sample)
        for p in peaks:
            if p - half < 0 or p + half > len(sig): continue
            win = sig[p - half: p + half]
            bp = bandpass_filter(win, fs=fs)
            pre = norm.observe(bp)
            if pre is None: continue
            with torch.no_grad():
                recon = model(torch.from_numpy(pre.astype(np.float32))
                              .unsqueeze(0)).squeeze(0).numpy()
            res = float(np.mean((pre - recon) ** 2))
            # Center beat = closest annotation within 100 ms of detected peak
            diffs = np.abs(ann_samples - p)
            i_center = int(np.argmin(diffs))
            center_sym = ann.symbol[i_center] if diffs[i_center] <= int(0.1 * fs) else "?"
            # All annotated beats whose R-peak falls inside the window
            in_win = (ann_samples >= p - half) & (ann_samples < p + half)
            beats_in_window = sorted(
                [(int(s - p), ann.symbol[k])
                 for k, s in enumerate(ann_samples) if in_win[k]],
                key=lambda x: x[0],
            )
            raw_mv = win.copy()
            out.append((rid, p, res, center_sym, pre, recon, raw_mv, fs,
                        beats_in_window))
        print(f"  {rid}: {sum(1 for r in out if r[0]==rid)} beats")
    return out


def pick_examples(beats, threshold):
    """For each target symbol, pick a mix of MISSED (FN), CAUGHT (TP),
    or just present beats - whatever is most informative."""
    by_sym: dict = defaultdict(list)
    for row in beats:
        sym = row[3]
        if sym in TARGETS or sym == "N":
            by_sym[sym].append(row)
    picked: list = []
    rng = np.random.default_rng(0)
    for sym, target_n in TARGETS.items():
        candidates = by_sym.get(sym, [])
        if not candidates:
            continue
        if sym == "N":
            # 8 true negatives + 2 false positives (rare normals near threshold)
            tn = sorted([c for c in candidates if c[2] <= threshold],
                        key=lambda r: r[2])
            fp = sorted([c for c in candidates if c[2] > threshold],
                        key=lambda r: -r[2])
            picked += [(*r, "TN") for r in tn[:8]]
            picked += [(*r, "FP") for r in fp[:2]]
        else:
            # Mix of catches (TP) and misses (FN), prioritising both edges
            tp = sorted([c for c in candidates if c[2] > threshold],
                        key=lambda r: -r[2])
            fn = sorted([c for c in candidates if c[2] <= threshold],
                        key=lambda r: -r[2])
            half = target_n // 2
            picked_tp = tp[:half]
            picked_fn = fn[:target_n - half]
            picked += [(*r, "TP") for r in picked_tp]
            picked += [(*r, "FN") for r in picked_fn]
    return picked


def render(picked, threshold, out_orig, out_recon, out_raw, fs=360):
    out_orig.parent.mkdir(parents=True, exist_ok=True)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    for row in picked:
        counts[row[3]] += 1

    cover_text = (
        f"{len(picked)} example beats, grouped by MIT-BIH annotation type\n\n"
        f"Threshold (model decision): {threshold}\n\n"
        f"Distribution:\n"
    )
    for sym, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        cover_text += f"  {sym} {SYMBOL_NAMES.get(sym, '?'):32} {n}\n"
    cover_text += (
        "\nLabels on each page:\n"
        "  Symbol      symbol of the centre beat (the one that labels the window)\n"
        "  Type        plain-language meaning\n"
        "  Source      MIT-BIH record + sample index (reproducible)\n"
        "  In window   ALL annotated beats inside this 2-second window,\n"
        "              listed as time-from-centre:symbol  (e.g. -380ms:N  *0ms:V*  +420ms:N)\n"
        "              Asterisks mark the centre beat - the one whose label is the\n"
        "              window label.  Other beats also affect the residual but their\n"
        "              labels are NOT used.  This is the per-beat-label limitation.\n"
        "  Residual    model's reconstruction error on the window\n"
        "  Threshold   the F1-optimal value 0.0434\n"
        "  Decision    model's NORMAL vs ANOMALY call\n"
        "  Outcome     TP, FN, TN, FP\n\n"
        "On the plot itself: short vertical ticks at the bottom mark every annotated\n"
        "R-peak in the window.  Green = NORMAL beat, magenta = ABNORMAL beat.\n"
        "The centre beat's symbol is shown in *asterisks* in the textual list.\n"
    )

    def render_one(pdf, row, mode: str):
        """mode: 'orig' (preprocessed only), 'recon' (preprocessed + recon),
                 'raw' (raw mV signal, no preprocessing)"""
        rid, p, res, sym, pre, recon, raw_mv, beat_fs, beats_in_window, outcome = row
        decision = "ANOMALY" if res > threshold else "NORMAL"
        truth = "ANOMALY" if sym in ANOMALY_SYMBOLS else "NORMAL"
        agree = decision == truth
        signal_to_plot = raw_mv if mode == "raw" else pre
        t = np.arange(len(signal_to_plot)) / beat_fs

        fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]
        if mode == "raw":
            ax.plot(t, raw_mv, color="#000000", lw=1.2, label="Raw ECG (mV)")
            ax.set_ylabel("amplitude (mV, raw)")
        else:
            ax.plot(t, pre, color="#1f77b4", lw=1.2,
                    label="Original (preprocessed)")
            ax.set_ylabel("amplitude (z-scored)")
            if mode == "recon":
                ax.plot(t, recon, color="#ff7f0e", lw=1.2, alpha=0.85,
                        label="Model reconstruction")
                ax.fill_between(t, pre, recon, color="#d62728", alpha=0.18,
                                label="Residual region")
        ax.set_xlabel("time (s)")
        ax.set_title(
            f"{outcome}: {SYMBOL_NAMES.get(sym, '?')} ({sym})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        # Mark every annotated R-peak inside the window with a vertical tick
        # and the beat symbol just above the x-axis. Box the centre beat.
        ymin, ymax = ax.get_ylim()
        center_offset = min(beats_in_window, key=lambda b: abs(b[0]))[0] \
            if beats_in_window else None
        for offs, s in beats_in_window:
            t_pos = (offs + len(pre) // 2) / beat_fs
            tick_color = "magenta" if s not in NORMAL_SYMBOLS else "green"
            ax.axvline(t_pos, ymin=0, ymax=0.06, color=tick_color, lw=2)
            label = f"*{s}*" if offs == center_offset else s
            ax.text(t_pos, ymin + (ymax - ymin) * 0.04, label,
                    color=tick_color, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax = axes[1]; ax.axis("off")
        in_win_str = "  ".join(
            f"{(offs/beat_fs)*1000:+5.0f}ms:{('*'+s+'*') if offs == center_offset else s}"
            for offs, s in beats_in_window
        ) if beats_in_window else "(none)"
        info = [
            f"Symbol      {sym}  (centre beat - the one that labels the window)",
            f"Type        {SYMBOL_NAMES.get(sym, '?')}",
            f"Source      record {rid}, sample {p}",
            f"In window   {in_win_str}",
            f"Residual    {res:.4f}    Threshold {threshold}",
            f"Decision    {decision}    (truth: {truth})",
            f"Outcome     {outcome}    {'agree' if agree else 'mismatch'}",
        ]
        ax.text(0.02, 0.95, "\n".join(info), va="top", ha="left",
                fontfamily="monospace", fontsize=11, transform=ax.transAxes)
        decision_color = "red" if decision == "ANOMALY" else "green"
        ax.text(0.98, 0.5, decision, va="center", ha="right",
                color="white", fontsize=22, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6",
                          facecolor=decision_color, edgecolor="none"),
                transform=ax.transAxes)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # Group by symbol for ordering, then by outcome
    sym_order = ["N", "V", "F", "A", "L", "R", "/", "f"]
    picked_sorted = sorted(picked, key=lambda r: (
        sym_order.index(r[3]) if r[3] in sym_order else 99,
        r[9],  # outcome string (last field after rid,p,res,sym,pre,recon,raw,fs,beats)
    ))

    targets = [
        (out_orig,  "orig",  "ECG examples - PREPROCESSED (z-scored), no reconstruction"),
        (out_recon, "recon", "ECG examples - WITH MODEL RECONSTRUCTION OVERLAID"),
        (out_raw,   "raw",   "ECG examples - RAW (mV), as it would come off a Holter"),
    ]
    for outpath, mode, title in targets:
        with PdfPages(outpath) as pdf:
            cover = plt.figure(figsize=(10, 7))
            cover.text(0.5, 0.93, title,
                       ha="center", fontsize=14, fontweight="bold")
            cover.text(0.05, 0.88, cover_text, va="top",
                       fontfamily="monospace", fontsize=10)
            pdf.savefig(cover); plt.close(cover)

            current_sym = None
            for row in picked_sorted:
                sym = row[3]
                if sym != current_sym:
                    sec = plt.figure(figsize=(10, 4))
                    sec.text(0.5, 0.55,
                             f"Section: {sym} - {SYMBOL_NAMES.get(sym, '?')}",
                             ha="center", va="center",
                             fontsize=18, fontweight="bold")
                    pdf.savefig(sec); plt.close(sec)
                    current_sym = sym
                render_one(pdf, row, mode)
        print(f"Saved {outpath}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    p.add_argument("--threshold", type=float, default=0.0434)
    p.add_argument("--warmup", type=int, default=10800)
    p.add_argument("--out-orig", default="docs/clinical_proof/examples_by_type.pdf")
    p.add_argument("--out-recon",
                   default="docs/clinical_proof/examples_by_type_recon.pdf")
    p.add_argument("--out-raw",
                   default="docs/clinical_proof/examples_by_type_raw.pdf")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    print("Loading model...")
    model = load_model(Path(args.model), ConvAutoencoder)
    print(f"Scoring beats from {len(RECORDS)} records...")
    beats = collect_beats(data_dir, model, args.threshold, args.warmup)
    print(f"Collected {len(beats)} beats total")

    picked = pick_examples(beats, args.threshold)
    print(f"\nPicked {len(picked)} example beats:")
    counts = defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for row in picked:
        counts[row[3]][row[9]] += 1
    for sym, c in counts.items():
        flat = " ".join(f"{k}={v}" for k, v in c.items() if v)
        print(f"  {sym} ({SYMBOL_NAMES.get(sym, '?'):30}): {flat}")

    render(picked, args.threshold,
           Path(args.out_orig), Path(args.out_recon), Path(args.out_raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
