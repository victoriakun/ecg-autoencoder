"""Verify how many annotated beats become the centre of their own
2-second window in (a) the batch dataset, (b) the real-time pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wfdb
from wfdb.processing import XQRS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="208")
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--window", type=int, default=720)
    p.add_argument("--tolerance-ms", type=float, default=50.0)
    args = p.parse_args()

    rec_path = str(Path(args.data_dir) / args.record)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    fs = int(rec.fs)
    sig_len = rec.p_signal.shape[0]
    half = args.window // 2
    tol = int(args.tolerance_ms / 1000.0 * fs)

    ann_samples = np.asarray(ann.sample)
    n_total = len(ann_samples)

    # (a) Batch coverage: every beat gets a window unless too close to edge
    has_batch_window = (ann_samples >= half) & (ann_samples + half <= sig_len)
    n_batch = int(has_batch_window.sum())

    # (b) Real-time coverage via XQRS detection
    print(f"Running XQRS on record {args.record}...")
    sig = rec.p_signal[:, 0].astype(float)
    xqrs = XQRS(sig=sig, fs=fs); xqrs.detect(verbose=False)
    detected = np.asarray(xqrs.qrs_inds)
    has_runtime_window = []
    for s in ann_samples:
        if s < half or s + half > sig_len:
            has_runtime_window.append(False); continue
        if len(detected) == 0:
            has_runtime_window.append(False); continue
        nearest = int(np.argmin(np.abs(detected - s)))
        has_runtime_window.append(abs(detected[nearest] - s) <= tol)
    has_runtime_window = np.asarray(has_runtime_window)
    n_runtime = int(has_runtime_window.sum())

    print(f"\nRecord {args.record}: {n_total} annotated beats")
    print(f"  BATCH coverage:   {n_batch:5}  ({n_batch/n_total*100:5.1f}%)  -- only edge beats dropped")
    print(f"  RUNTIME coverage: {n_runtime:5}  ({n_runtime/n_total*100:5.1f}%)  -- XQRS within +/- {args.tolerance_ms:.0f} ms")
    missed = n_total - n_runtime
    print(f"  Missed at runtime: {missed} ({missed/n_total*100:.1f}%)")
    if missed:
        miss_idx = np.where(~has_runtime_window)[0]
        miss_syms = [ann.symbol[i] for i in miss_idx]
        from collections import Counter
        c = Counter(miss_syms)
        print(f"  Symbols of missed beats: {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
