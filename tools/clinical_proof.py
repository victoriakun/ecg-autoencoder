"""Generate a multi-page PDF of example beats for cardiologist review.

For each beat-centered window, runs the model end-to-end (real-time
preprocessing path) and classifies it against the calibrated threshold.
Picks the most informative examples in each confusion-matrix cell:

- True positive  : abnormal beat correctly flagged (high residual)
- False negative : abnormal beat missed by the model (residual just below thr)
- False positive : normal beat wrongly flagged (highest-residual normals)
- True negative  : normal beat correctly cleared (clean reconstructions)

Each page shows the 2-second ECG window with the model's reconstruction
overlaid plus a clinical label box (ground truth symbol, predicted class,
residual, threshold, decision).

Output: docs/clinical_proof/clinical_proof.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dataset import NORMAL_SYMBOLS, ANOMALY_SYMBOLS, MITBIH_RECORDS
from models import ConvAutoencoder
from preprocess import preprocess
from realtime.inference import load_model


SYMBOL_MEANINGS = {
    "N": "Normal beat",
    "L": "Left bundle branch block",
    "R": "Right bundle branch block",
    "A": "Atrial premature beat",
    "a": "Aberrated atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature beat",
    "V": "Premature ventricular contraction (PVC)",
    "F": "Fusion of ventricular and normal",
    "!": "Ventricular flutter wave",
    "e": "Atrial escape beat",
    "j": "Nodal (junctional) escape beat",
    "E": "Ventricular escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal",
    "Q": "Unclassifiable",
}


def extract_records(records, data_dir, window_samples=720, lead=0):
    """Extract beat-centered windows using the ORIGINAL preprocessing
    methodology from dataset.py / evaluate.py: bandpass + zero-mean /
    unit-variance applied ONCE across the whole record, then cut
    beat-centered windows. This matches the residual scale at which
    the published threshold 0.0434 was F1-optimal."""
    xs, ys, syms, sources, raws = [], [], [], [], []
    for rid in records:
        path = str(data_dir / rid)
        try:
            rec = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, "atr")
        except Exception as e:
            print(f"  skip {rid}: {e}")
            continue
        raw_signal = rec.p_signal[:, lead].astype(float)
        fs = rec.fs
        signal = preprocess(raw_signal, fs, apply_bandpass=True, apply_notch=False)
        half = window_samples // 2
        for idx, sym in zip(ann.sample, ann.symbol):
            if idx - half < 0 or idx + half > len(signal):
                continue
            if sym in NORMAL_SYMBOLS:
                y = 0
            elif sym in ANOMALY_SYMBOLS:
                y = 1
            else:
                continue
            window = signal[idx - half:idx + half]
            raw_window = raw_signal[idx - half:idx + half]
            if len(window) != window_samples:
                continue
            xs.append(window.astype(np.float32))
            raws.append(raw_window.astype(np.float32))
            ys.append(y)
            syms.append(sym)
            sources.append(f"rec {rid}, sample {idx}")
    return (np.asarray(xs), np.asarray(ys), syms, sources, np.asarray(raws))


def _robust_mv_ylim(ax, win):
    """Set y-limits using 1st-99th percentile so a single artefact spike
    doesn't squash the visible trace. Minimum visible span 1.5 mV."""
    finite = win[np.isfinite(win)]
    if finite.size:
        p_lo = float(np.percentile(finite, 1))
        p_hi = float(np.percentile(finite, 99))
    else:
        p_lo, p_hi = -0.75, 0.75
    span = max(p_hi - p_lo, 1.5)
    mid = 0.5 * (p_lo + p_hi)
    ax.set_ylim(mid - span * 0.7, mid + span * 0.7)


def render_blind_page(pdf, win, beat_id, fs=360):
    """Render a RAW-mV ECG window on standard pink ECG-paper grid -
    cardiologist marks judgment (no labels visible)."""
    t = np.arange(len(win)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={"height_ratios": [4, 1]})
    ax = axes[0]
    ax.plot(t, win, color="#000000", lw=1.4)
    ax.set_xlim(0, len(win) / fs)
    _robust_mv_ylim(ax, win)
    for x in np.arange(0, len(win) / fs + 1e-6, 0.04):
        ax.axvline(x, color="#f7c4c4", lw=0.4, zorder=0)
    for x in np.arange(0, len(win) / fs + 1e-6, 0.20):
        ax.axvline(x, color="#e07b7b", lw=0.8, zorder=0)
    y0 = np.floor(ax.get_ylim()[0] * 10) / 10
    y1 = np.ceil(ax.get_ylim()[1] * 10) / 10
    for y in np.arange(y0, y1 + 1e-6, 0.1):
        ax.axhline(y, color="#f7c4c4", lw=0.4, zorder=0)
    for y in np.arange(y0, y1 + 1e-6, 0.5):
        ax.axhline(y, color="#e07b7b", lw=0.8, zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (mV)")
    ax.set_title(f"Beat #{beat_id} — judge in isolation",
                 fontsize=13, fontweight="bold")

    ax = axes[1]
    ax.axis("off")
    ax.text(0.02, 0.85, f"Beat #{beat_id}",
            fontfamily="monospace", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.02, 0.55,
            "Your judgment:   [ ] NORMAL     [ ] ABNORMAL     [ ] UNREADABLE",
            fontfamily="monospace", fontsize=11, transform=ax.transAxes)
    ax.text(0.02, 0.20, "If abnormal — best guess of type:  __________________________",
            fontfamily="monospace", fontsize=10, transform=ax.transAxes,
            color="#444")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def render_page(pdf, win, recon, residual, threshold, label, sym, source,
                pred_label, fs=360):
    t = np.arange(len(win)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(t, win, color="#1f77b4", lw=1.2, label="Original (preprocessed)")
    ax.plot(t, recon, color="#ff7f0e", lw=1.2, alpha=0.85, label="Reconstruction")
    ax.fill_between(t, win, recon, color="#d62728", alpha=0.18,
                    label="Residual region")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (z-scored)")
    ax.set_title(f"{label}: ground truth = {SYMBOL_MEANINGS.get(sym, sym)} ({sym})",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    decision_color = "red" if pred_label == "ANOMALY" else "green"
    info_lines = [
        f"Source       : {source}",
        f"Beat symbol  : {sym}  ({SYMBOL_MEANINGS.get(sym, 'unknown')})",
        f"Ground truth : {'ANOMALY' if sym not in NORMAL_SYMBOLS else 'NORMAL'}",
        f"Residual     : {residual:.4f}",
        f"Threshold    : {threshold:.4f}",
        f"Model says   : {pred_label}",
    ]
    ax.axis("off")
    ax.text(0.02, 0.95, "\n".join(info_lines), va="top", ha="left",
            fontfamily="monospace", fontsize=11, transform=ax.transAxes)
    ax.text(0.98, 0.5, pred_label, va="center", ha="right",
            color="white", fontsize=22, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=decision_color,
                      edgecolor="none"),
            transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pick(idx_list, residuals, n, mode):
    if not len(idx_list):
        return []
    sub = sorted(idx_list, key=lambda i: residuals[i],
                 reverse=(mode == "highest"))
    return sub[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    p.add_argument("--calibration", default="models/mitbih_autoencoder_best.batchscale.calibration.json")
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--records", default=",".join(MITBIH_RECORDS[:8]))
    p.add_argument("--per-cell", type=int, default=12,
                   help="examples per confusion-matrix cell "
                        "(4 cells -> 48 beats by default; needed for a "
                        "kappa-usable blind worksheet)")
    p.add_argument("--out", default="docs/clinical_proof/clinical_proof.pdf")
    p.add_argument("--blind", action="store_true",
                   help="generate a blind version: hides the model's decision "
                        "and ground-truth so the cardiologist can label first, "
                        "then we compare with model + MIT-BIH labels afterwards")
    p.add_argument("--blind-out", default="docs/clinical_proof/clinical_blind.pdf")
    p.add_argument("--blind-model-out",
                   default="docs/clinical_proof/clinical_blind_model_results.pdf",
                   help="companion PDF showing the model's decision for each "
                        "blind beat in the SAME blind order, for post-meeting "
                        "discussion of each disagreement")
    p.add_argument("--answer-key-out",
                   default="docs/clinical_proof/blind_answer_key.csv")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    threshold = float(json.loads(Path(args.calibration).read_text())["p99"])
    print(f"Using F1-optimal threshold = {threshold:.4f}")

    print("Loading model and extracting beat-centered windows...")
    model = load_model(Path(args.model), ConvAutoencoder)
    X, y, syms, sources, X_raw = extract_records(args.records.split(","), data_dir)
    print(f"  {len(X)} windows, {y.sum()} anomalies")

    print("Computing reconstructions and residuals...")
    recons = []
    residuals = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.from_numpy(X[i:i + 256])
            recon = model(batch).numpy()
            recons.append(recon)
            residuals.extend(((X[i:i + 256] - recon) ** 2).mean(axis=1).tolist())
    recons = np.concatenate(recons, axis=0)
    residuals = np.asarray(residuals)
    preds = (residuals > threshold).astype(int)

    tp = [i for i in range(len(y)) if y[i] == 1 and preds[i] == 1]
    fn = [i for i in range(len(y)) if y[i] == 1 and preds[i] == 0]
    fp = [i for i in range(len(y)) if y[i] == 0 and preds[i] == 1]
    tn = [i for i in range(len(y)) if y[i] == 0 and preds[i] == 0]
    print(f"Confusion: TP={len(tp)}  FN={len(fn)}  FP={len(fp)}  TN={len(tn)}")
    print(f"  Sensitivity = {len(tp) / max(1, len(tp) + len(fn)):.3f}")
    print(f"  Specificity = {len(tn) / max(1, len(tn) + len(fp)):.3f}")

    selections = [
        ("TRUE POSITIVE — model correctly flagged abnormal",
         pick(tp, residuals, args.per_cell, "highest"), "ANOMALY"),
        ("FALSE NEGATIVE — model MISSED an abnormal beat",
         pick(fn, residuals, args.per_cell, "highest"), "NORMAL"),
        ("FALSE POSITIVE — model wrongly flagged a normal beat",
         pick(fp, residuals, args.per_cell, "highest"), "ANOMALY"),
        ("TRUE NEGATIVE — model correctly cleared a normal beat",
         pick(tn, residuals, args.per_cell, "lowest"), "NORMAL"),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        cover = plt.figure(figsize=(10, 6))
        cover.text(0.5, 0.85, "Clinical proof-of-concept",
                   ha="center", fontsize=22, fontweight="bold")
        cover.text(0.5, 0.78, "ECG anomaly detection — example beats",
                   ha="center", fontsize=14)
        summary = (
            f"Model: {Path(args.model).name}\n"
            f"Threshold (F1-optimal, real-time preprocessing): {threshold:.4f}\n"
            f"Records used: {args.records}\n"
            f"Total beats inspected: {len(y)}    Anomalies in set: {int(y.sum())}\n\n"
            f"Confusion matrix on these beats:\n"
            f"  True positives  : {len(tp)}    (correctly caught)\n"
            f"  False negatives : {len(fn)}    (missed by the model)\n"
            f"  False positives : {len(fp)}    (false alarms)\n"
            f"  True negatives  : {len(tn)}    (correctly cleared)\n\n"
            f"  Sensitivity = {len(tp) / max(1, len(tp) + len(fn)):.3f}\n"
            f"  Specificity = {len(tn) / max(1, len(tn) + len(fp)):.3f}\n\n"
            f"How to read the next pages: each page shows one 2-second window\n"
            f"with the original ECG (blue) and the model's reconstruction (orange).\n"
            f"The shaded red area between them is the reconstruction error.\n"
            f"A beat is flagged when that error exceeds the threshold above."
        )
        cover.text(0.08, 0.55, summary, va="top", fontfamily="monospace",
                   fontsize=10)
        pdf.savefig(cover)
        plt.close(cover)

        for label, idx_list, _pred in selections:
            section = plt.figure(figsize=(10, 4))
            section.text(0.5, 0.5, label, ha="center", va="center",
                         fontsize=18, fontweight="bold", wrap=True)
            section.text(0.5, 0.3, f"({len(idx_list)} example(s) follow)",
                         ha="center", fontsize=11, color="grey")
            pdf.savefig(section)
            plt.close(section)

            for i in idx_list:
                pred_label = "ANOMALY" if preds[i] == 1 else "NORMAL"
                render_page(pdf, X[i], recons[i], residuals[i], threshold,
                            label, syms[i], sources[i], pred_label)

    print(f"\nSaved {out_path}")

    # Optionally also render a blind version of the same selected beats,
    # plus a CSV answer key so you can score her labels later.
    if args.blind:
        all_picked = []
        for label, idx_list, _ in selections:
            all_picked.extend(idx_list)
        rng = np.random.default_rng(0)
        order = list(rng.permutation(len(all_picked)))
        blind_path = Path(args.blind_out)
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(blind_path) as pdf:
            cover = plt.figure(figsize=(10, 6))
            cover.text(0.5, 0.85, "Blind labeling — cardiologist worksheet",
                       ha="center", fontsize=20, fontweight="bold")
            cover.text(0.5, 0.78,
                       f"{len(all_picked)} beats, randomized order",
                       ha="center", fontsize=12)
            cover.text(0.08, 0.65,
                       "Instructions:\n"
                       "  1. For each beat, mark NORMAL / ABNORMAL / UNREADABLE.\n"
                       "  2. If abnormal, optionally jot the suspected beat type.\n"
                       "  3. DO NOT confer with anyone or look at the model output.\n"
                       "  4. Return the marked sheets - we'll compare with the\n"
                       "     model's decisions and the MIT-BIH reference labels.\n\n"
                       "We will compute Cohen's kappa between your labels, the\n"
                       "model's labels, and the MIT-BIH reference labels - this\n"
                       "is the rigorous validation step.",
                       va="top", fontfamily="monospace", fontsize=11)
            pdf.savefig(cover); plt.close(cover)
            for blind_id, original_idx in enumerate(order, start=1):
                i = all_picked[original_idx]
                render_blind_page(pdf, X_raw[i], blind_id)

        # Write answer key CSV
        import csv
        key_path = Path(args.answer_key_out)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with key_path.open("w") as f:
            w = csv.writer(f)
            w.writerow(["beat_id", "source", "ground_truth_symbol",
                        "ground_truth_class", "model_residual",
                        "model_threshold", "model_decision"])
            for blind_id, original_idx in enumerate(order, start=1):
                i = all_picked[original_idx]
                w.writerow([
                    blind_id, sources[i], syms[i],
                    "ANOMALY" if y[i] == 1 else "NORMAL",
                    f"{residuals[i]:.4f}", f"{threshold:.4f}",
                    "ANOMALY" if preds[i] == 1 else "NORMAL",
                ])
        print(f"Saved {blind_path}")
        print(f"Saved {key_path}  (do NOT show to the cardiologist)")

        # Companion PDF: same beats, same blind order, with the model's
        # reconstruction overlay + decision shown. For walking through
        # disagreements together AFTER she has labelled the blind sheet.
        companion_path = Path(args.blind_model_out)
        companion_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(companion_path) as pdf:
            cover = plt.figure(figsize=(10, 6))
            cover.text(0.5, 0.85,
                       "Blind worksheet — model results companion",
                       ha="center", fontsize=20, fontweight="bold")
            cover.text(0.5, 0.78,
                       f"{len(all_picked)} beats, SAME order as "
                       f"clinical_blind.pdf",
                       ha="center", fontsize=12)
            cover.text(0.08, 0.65,
                       "Use AFTER the cardiologist has marked her sheet.\n\n"
                       "Each page corresponds to the matching beat number\n"
                       "in clinical_blind.pdf and shows:\n"
                       "  - the same 2-second window in z-scored space\n"
                       "  - the model's reconstruction (orange overlay)\n"
                       "  - the residual region (red shaded)\n"
                       "  - the MIT-BIH ground-truth symbol + class\n"
                       "  - the model's residual, threshold, and decision\n\n"
                       "Outcome at top of each page:\n"
                       "  TP = model caught a true anomaly\n"
                       "  FN = model missed an anomaly\n"
                       "  FP = model wrongly flagged a normal beat\n"
                       "  TN = model correctly cleared a normal beat\n\n"
                       "Walk through together; pause on every disagreement\n"
                       "between her label and the model's decision.",
                       va="top", fontfamily="monospace", fontsize=11)
            pdf.savefig(cover); plt.close(cover)
            for blind_id, original_idx in enumerate(order, start=1):
                i = all_picked[original_idx]
                truth_label = "ANOMALY" if y[i] == 1 else "NORMAL"
                pred_label = "ANOMALY" if preds[i] == 1 else "NORMAL"
                if truth_label == "ANOMALY" and pred_label == "ANOMALY":
                    outcome = "TP"
                elif truth_label == "ANOMALY" and pred_label == "NORMAL":
                    outcome = "FN"
                elif truth_label == "NORMAL" and pred_label == "ANOMALY":
                    outcome = "FP"
                else:
                    outcome = "TN"
                page_label = f"Beat #{blind_id}  -  {outcome}"
                render_page(pdf, X[i], recons[i], residuals[i], threshold,
                            page_label, syms[i], sources[i], pred_label)
        print(f"Saved {companion_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
