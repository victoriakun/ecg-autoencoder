"""For one MIT-BIH record, run the Option-C runtime pipeline and compare
each detected beat's model decision to the cardiologist-annotated label.

Outputs:
1. Console: confusion matrix overall + per-beat-symbol breakdown
2. PDF page(s) showing the ECG timeline with:
   - GREEN bars under each annotated NORMAL beat
   - GREEN-DOTTED bars under each annotated ABNORMAL beat
   - RED shading on the windows the MODEL flagged as anomaly
   You can then visually see where they agree/disagree.

Usage:
    PYTHONPATH=. python tools/compare_to_annotation.py --record 208
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wfdb
from wfdb.processing import XQRS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from dataset import NORMAL_SYMBOLS, ANOMALY_SYMBOLS
from models import ConvAutoencoder
from preprocess import bandpass_filter
from realtime.inference import load_model
from realtime.normalizer import WarmupNormalizer


SYMBOL_NAMES = {
    "N": "Normal", "L": "LBBB", "R": "RBBB", "A": "Atrial premature",
    "a": "Aberrated atrial", "J": "Nodal premature",
    "S": "Supraventricular", "V": "PVC (ventricular ectopic)",
    "F": "Fusion (V+N)", "!": "Ventricular flutter",
    "e": "Atrial escape", "j": "Nodal escape",
    "E": "Ventricular escape", "/": "Paced", "f": "Fusion (P+N)",
    "Q": "Unclassifiable",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="208")
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    p.add_argument("--threshold", type=float, default=0.0434)
    p.add_argument("--warmup-samples", type=int, default=10800)
    p.add_argument("--out", default=None,
                   help="output PDF (default: docs/clinical_proof/compare_<rec>.pdf)")
    p.add_argument("--page-seconds", type=float, default=20.0,
                   help="seconds per timeline page")
    p.add_argument("--max-pages", type=int, default=6)
    args = p.parse_args()

    rec_path = str(Path(args.data_dir) / args.record)
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal[:, 0].astype(float)
    fs = int(rec.fs)
    ann = wfdb.rdann(rec_path, "atr")

    print(f"Record {args.record}: {len(sig)/fs/60:.1f} min @ {fs} Hz, "
          f"{len(ann.sample)} annotated beats")

    # --- Run the runtime pipeline (Option C) ---
    print("Detecting R-peaks with XQRS...")
    xqrs = XQRS(sig=sig, fs=fs); xqrs.detect(verbose=False)
    detected = list(xqrs.qrs_inds)

    print("Loading model and computing residuals on beat-centered windows...")
    model = load_model(Path(args.model), ConvAutoencoder)
    norm = WarmupNormalizer(args.warmup_samples)
    WIN = 720; half = WIN // 2

    rows = []
    for p in detected:
        if p - half < 0 or p + half > len(sig):
            continue
        win = sig[p - half: p + half]
        bp = bandpass_filter(win, fs=fs)
        pre = norm.observe(bp)
        if pre is None:
            continue
        with torch.no_grad():
            recon = model(torch.from_numpy(pre.astype(np.float32))
                          .unsqueeze(0)).squeeze(0).numpy()
        residual = float(np.mean((pre - recon) ** 2))
        # Find the closest annotation (within 100 ms)
        diffs = np.abs(np.asarray(ann.sample) - p)
        nearest = int(np.argmin(diffs))
        if diffs[nearest] <= int(0.1 * fs):
            sym = ann.symbol[nearest]
        else:
            sym = "?"  # detected but no annotation
        rows.append((p, residual, sym))

    print(f"Scored {len(rows)} beats (skipped first ~{args.warmup_samples/fs:.0f} s for warmup).")

    # --- Confusion matrix ---
    pred_anom = np.asarray([r[1] > args.threshold for r in rows])
    true_anom = np.asarray([r[2] in ANOMALY_SYMBOLS for r in rows])
    has_label = np.asarray([r[2] != "?" for r in rows])
    pa, ta = pred_anom[has_label], true_anom[has_label]
    tp = int((pa & ta).sum()); fn = int((~pa & ta).sum())
    fp = int((pa & ~ta).sum()); tn = int((~pa & ~ta).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp); f1 = 2 * prec * sens / max(1e-8, prec + sens)
    print(f"\nConfusion matrix (beat-level on {ta.size} matched beats, threshold={args.threshold}):")
    print(f"           model NORMAL   model ANOMALY")
    print(f"  TRUE N  : {tn:>10d}    {fp:>10d}    (FP rate {fp/max(1,fp+tn)*100:.1f}%)")
    print(f"  TRUE A  : {fn:>10d}    {tp:>10d}    (recall  {sens*100:.1f}%)")
    print(f"  -> Sensitivity {sens:.3f}, Specificity {spec:.3f}, "
          f"Precision {prec:.3f}, F1 {f1:.3f}")

    # --- Per-symbol breakdown ---
    print("\nPer-beat-type breakdown:")
    by_sym: dict = {}
    for (_, res, sym) in rows:
        if sym == "?": continue
        by_sym.setdefault(sym, []).append(res > args.threshold)
    print(f"  {'sym':>4} {'meaning':30}  {'count':>5}  {'flagged':>8}  {'flag_rate':>9}")
    for sym in sorted(by_sym, key=lambda s: -len(by_sym[s])):
        arr = np.asarray(by_sym[sym])
        meaning = SYMBOL_NAMES.get(sym, "?")
        print(f"  {sym:>4} {meaning:30}  {len(arr):>5}  {int(arr.sum()):>8}  "
              f"{arr.mean()*100:>8.1f}%")

    # --- Timeline PDF ---
    out_path = Path(args.out or f"docs/clinical_proof/compare_{args.record}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_n = int(args.page_seconds * fs)
    n_pages = min(args.max_pages, (len(sig) - args.warmup_samples) // page_n + 1)

    sig_bp = bandpass_filter(sig, fs=fs)
    print(f"\nWriting timeline PDF ({n_pages} pages of {args.page_seconds}s each) -> {out_path}")
    with PdfPages(out_path) as pdf:
        cover = plt.figure(figsize=(11, 8))
        cover.suptitle(f"Model vs cardiologist labels — record {args.record}",
                       fontsize=16, fontweight="bold")
        text = (
            f"Threshold: {args.threshold}\n"
            f"Beats scored: {ta.size}\n\n"
            f"Confusion matrix:\n"
            f"  TP (model+annot agree abnormal): {tp}\n"
            f"  TN (both agree normal):          {tn}\n"
            f"  FP (model says abn, annot N):    {fp}\n"
            f"  FN (annot abn, model says N):    {fn}\n\n"
            f"Sensitivity (recall): {sens:.3f}\n"
            f"Specificity:          {spec:.3f}\n"
            f"Precision:            {prec:.3f}\n"
            f"F1 score:             {f1:.3f}\n\n"
            "Per-beat-type breakdown:\n"
        )
        for sym in sorted(by_sym, key=lambda s: -len(by_sym[s])):
            arr = np.asarray(by_sym[sym])
            text += f"  {sym} {SYMBOL_NAMES.get(sym,'?'):28} {len(arr):4}  flagged {arr.mean()*100:5.1f}%\n"
        text += (
            "\nLegend on next pages:\n"
            "  Blue line:    bandpass-filtered ECG\n"
            "  Green tick:   annotated NORMAL beat (cardiologist)\n"
            "  Magenta tick: annotated ABNORMAL beat (cardiologist)\n"
            "  Red shading:  beat the MODEL flagged as anomaly\n"
            "An ideal model has red shading exactly under magenta ticks."
        )
        cover.text(0.05, 0.92, text, va="top", fontfamily="monospace", fontsize=10)
        pdf.savefig(cover); plt.close(cover)

        for page in range(n_pages):
            t0 = args.warmup_samples + page * page_n
            t1 = min(len(sig), t0 + page_n)
            t = np.arange(t0, t1) / fs
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(t, sig_bp[t0:t1], color="#1f77b4", lw=0.8)
            ax.set_xlim(t0 / fs, t1 / fs)
            ymin, ymax = ax.get_ylim()
            # True annotations
            for s, sym in zip(ann.sample, ann.symbol):
                if s < t0 or s >= t1: continue
                color = "green" if sym in NORMAL_SYMBOLS else "magenta"
                ax.axvline(s / fs, ymin=0, ymax=0.05, color=color, lw=1.5)
            # Model flags (red shading on flagged beat windows)
            for (p, res, sym) in rows:
                if p < t0 or p >= t1: continue
                if res > args.threshold:
                    ax.add_patch(Rectangle(
                        ((p - half) / fs, ymin), WIN / fs, ymax - ymin,
                        color="red", alpha=0.12, lw=0,
                    ))
            ax.set_xlabel("time (s)")
            ax.set_ylabel("ECG (band-passed)")
            ax.set_title(f"Record {args.record} — t={t0/fs:.0f}-{t1/fs:.0f} s")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
