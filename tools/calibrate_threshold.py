"""Compute a fixed-threshold calibration JSON for a real-time model checkpoint.

Runs the given checkpoint on a mostly-normal MIT-BIH record, collects the
residual of every overlapping 2-s window, and saves the 99th percentile as
the base threshold used by `FixedOnlineThreshold`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wfdb

from models import ConvAutoencoder
from preprocess import preprocess
from realtime.inference import load_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to checkpoint .pt")
    p.add_argument("--record", default="data/mitbih/100",
                   help="MIT-BIH record path (without extension)")
    p.add_argument("--out", required=True, help="output calibration JSON")
    p.add_argument("--window", type=int, default=720)
    p.add_argument("--stride", type=int, default=180)
    p.add_argument("--fs", type=int, default=360)
    args = p.parse_args()

    model = load_model(Path(args.model), ConvAutoencoder)
    rec = wfdb.rdrecord(args.record)
    signal = rec.p_signal[:, 0].astype(float)

    residuals: list[float] = []
    for start in range(0, signal.size - args.window + 1, args.stride):
        window = signal[start:start + args.window]
        pre = preprocess(window, fs=args.fs, apply_bandpass=True, apply_notch=False)
        x = torch.from_numpy(pre).float().unsqueeze(0)
        with torch.no_grad():
            recon = model(x).squeeze(0).numpy().astype(float)
        residuals.append(float(np.mean((pre - recon) ** 2)))

    arr = np.asarray(residuals)
    calib = {
        "source_record": args.record,
        "n_windows": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
