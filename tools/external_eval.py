"""External-database validation for the ECG anomaly detector.

Supports MIT-BIH NSRDB (all-normal sinus rhythm) and AFDB (atrial fibrillation
episode-labelled) at their native sampling rates. Resamples to 360 Hz, runs the
real-time-style pipeline (bandpass + warmup global normalize + per-stride
windows + model + threshold), and reports clinically meaningful stats.

Usage:
    PYTHONPATH=. python tools/external_eval.py --db nsrdb --records 16265,16272
    PYTHONPATH=. python tools/external_eval.py --db afdb --records 04015,04043
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

from models import ConvAutoencoder
from preprocess import bandpass_filter
from realtime.inference import load_model
from realtime.normalizer import WarmupNormalizer


TARGET_FS = 360
WINDOW_SAMPLES = 720
STRIDE_SAMPLES = 180
WARMUP_SAMPLES = 10800


def fetch(db: str, records: list[str], data_dir: Path) -> None:
    """Download records if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [r for r in records if not (data_dir / f"{r}.dat").exists()]
    if not missing:
        return
    print(f"Downloading {len(missing)} record(s) from PhysioNet '{db}'...")
    pn_name = {"nsrdb": "nsrdb", "afdb": "afdb"}[db]
    wfdb.dl_database(pn_name, str(data_dir), records=missing)


def resample(signal: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return signal
    g = np.gcd(src_fs, dst_fs)
    return resample_poly(signal, dst_fs // g, src_fs // g)


def run_record(model, record_path: str, db: str, max_seconds: float = 0) -> dict:
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal[:, 0].astype(float)
    src_fs = int(rec.fs)
    if max_seconds > 0:
        sig = sig[:int(max_seconds * src_fs)]
    sig = resample(sig, src_fs, TARGET_FS)

    af_mask = None
    if db == "afdb":
        try:
            ann = wfdb.rdann(record_path, "atr")
            af_mask = np.zeros(len(sig), dtype=bool)
            current_state = "(N"
            for sample, aux in zip(ann.sample, ann.aux_note):
                if aux:
                    current_state = aux.strip()
                # Project to resampled index
                resampled_idx = int(sample * TARGET_FS / src_fs)
                if resampled_idx < len(sig) and current_state == "(AFIB":
                    af_mask[resampled_idx:] = True
                elif resampled_idx < len(sig):
                    af_mask[resampled_idx:] = False
        except Exception:
            af_mask = None

    norm = WarmupNormalizer(WARMUP_SAMPLES)
    residuals = []
    in_af = []
    n_windows = (len(sig) - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1
    for i in range(n_windows):
        start = i * STRIDE_SAMPLES
        win = sig[start:start + WINDOW_SAMPLES]
        if win.size != WINDOW_SAMPLES:
            break
        bp = bandpass_filter(win, fs=TARGET_FS)
        pre = norm.observe(bp)
        if pre is None:
            continue
        with torch.no_grad():
            x = torch.from_numpy(pre.astype(np.float32)).unsqueeze(0)
            recon = model(x).squeeze(0).numpy().astype(float)
        residuals.append(float(np.mean((pre - recon) ** 2)))
        if af_mask is not None:
            in_af.append(bool(af_mask[start + WINDOW_SAMPLES // 2]))

    return {
        "record": Path(record_path).name,
        "src_fs": src_fs,
        "duration_sec": len(sig) / TARGET_FS,
        "n_windows_scored": len(residuals),
        "residuals": np.asarray(residuals),
        "in_af": np.asarray(in_af) if in_af else None,
    }


def adaptive_threshold(residuals: np.ndarray, q: float = 0.95) -> float:
    """Per-record adaptive threshold = q-th percentile of that record's
    residuals. Mirrors the percentile-mode dynamic threshold used in the
    real-time pipeline."""
    return float(np.quantile(residuals, q))


def report(rows: list[dict], threshold: float, db: str) -> dict:
    all_res = np.concatenate([r["residuals"] for r in rows])
    flagged = all_res > threshold

    # Adaptive (per-record) threshold metrics — useful to show the model
    # can separate within a record even when absolute scale shifts.
    adaptive_flagged = np.concatenate([
        r["residuals"] > adaptive_threshold(r["residuals"], 0.95) for r in rows
    ])
    summary = {
        "db": db,
        "threshold": threshold,
        "n_records": len(rows),
        "total_windows": int(all_res.size),
        "total_duration_min": float(sum(r["duration_sec"] for r in rows) / 60),
        "alert_rate_per_min": float(flagged.sum() / (sum(r["duration_sec"] for r in rows) / 60)),
        "fraction_flagged": float(flagged.mean()),
        "residual_p50": float(np.quantile(all_res, 0.50)),
        "residual_p95": float(np.quantile(all_res, 0.95)),
        "residual_p99": float(np.quantile(all_res, 0.99)),
    }

    if db == "nsrdb":
        summary["interpretation"] = (
            "All windows are healthy sinus rhythm. Every flag is a FALSE POSITIVE. "
            f"False positive rate = {summary['fraction_flagged'] * 100:.2f}%; "
            f"~{summary['alert_rate_per_min']:.2f} false alarms per minute per patient."
        )
    elif db == "afdb":
        all_af = np.concatenate(
            [r["in_af"] for r in rows if r["in_af"] is not None and r["in_af"].size]
        )
        if all_af.size:
            in_af = all_af
            tp = int((flagged & in_af).sum())
            fn = int((~flagged & in_af).sum())
            fp = int((flagged & ~in_af).sum())
            tn = int((~flagged & ~in_af).sum())
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            summary["AF_window_count"] = int(in_af.sum())
            summary["non_AF_window_count"] = int((~in_af).sum())
            summary["sensitivity_to_AF_fixed"] = sens
            summary["specificity_to_AF_fixed"] = spec

            tp_a = int((adaptive_flagged & in_af).sum())
            fn_a = int((~adaptive_flagged & in_af).sum())
            fp_a = int((adaptive_flagged & ~in_af).sum())
            tn_a = int((~adaptive_flagged & ~in_af).sum())
            sens_a = tp_a / max(1, tp_a + fn_a)
            spec_a = tn_a / max(1, tn_a + fp_a)
            summary["sensitivity_to_AF_adaptive_p95"] = sens_a
            summary["specificity_to_AF_adaptive_p95"] = spec_a
            summary["interpretation"] = (
                f"FIXED 0.043 threshold: caught {sens * 100:.1f}% of AF, "
                f"{(1 - spec) * 100:.1f}% non-AF false-positive rate. "
                f"ADAPTIVE per-record p95: caught {sens_a * 100:.1f}% of AF, "
                f"{(1 - spec_a) * 100:.1f}% non-AF false-positive rate."
            )
    elif db == "nsrdb":
        summary["fraction_flagged_adaptive_p95"] = float(adaptive_flagged.mean())
        summary["interpretation"] += (
            f"  ADAPTIVE per-record p95 threshold: {summary['fraction_flagged_adaptive_p95']*100:.2f}% flagged "
            "(by construction ~5% — this is what the dynamic mode would produce)."
        )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, choices=["nsrdb", "afdb"])
    p.add_argument("--records", required=True, help="comma-separated record IDs")
    p.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    p.add_argument("--threshold", type=float, default=0.04339778050780296,
                   help="default = checkpoint-stored F1-optimal threshold")
    p.add_argument("--out-json", default=None)
    p.add_argument("--max-seconds", type=float, default=0,
                   help="limit each record to first N seconds (0 = full)")
    args = p.parse_args()

    data_dir = Path(f"data/{args.db}")
    records = [r.strip() for r in args.records.split(",")]
    fetch(args.db, records, data_dir)

    print(f"Loading {args.model}...")
    model = load_model(Path(args.model), ConvAutoencoder)

    rows = []
    for rid in records:
        path = str(data_dir / rid)
        if not Path(f"{path}.dat").exists():
            print(f"  skip {rid}: file missing"); continue
        print(f"  scoring {rid}...")
        r = run_record(model, path, args.db, max_seconds=args.max_seconds)
        print(f"    {r['n_windows_scored']} windows, "
              f"residual median = {np.median(r['residuals']):.4f}")
        rows.append(r)

    summary = report(rows, args.threshold, args.db)
    # Drop arrays from per-record summaries before printing
    for r in rows:
        r["residuals"] = float(r["residuals"].mean())
        if r["in_af"] is not None:
            r["in_af"] = float(r["in_af"].mean())
    summary["per_record"] = rows
    print("\n" + json.dumps(summary, indent=2, default=str))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nSaved {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
