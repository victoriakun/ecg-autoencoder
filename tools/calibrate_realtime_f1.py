"""F1-optimal threshold calibrated with the REAL-TIME preprocessing path.

Differs from evaluate.py by applying `preprocess()` per window (not once on
the whole signal), so residuals are on the same scale the real-time pipeline
actually sees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wfdb
from sklearn.metrics import precision_recall_curve, roc_auc_score

from dataset import NORMAL_SYMBOLS, ANOMALY_SYMBOLS, MITBIH_RECORDS
from models import ConvAutoencoder
from preprocess import preprocess
from realtime.inference import load_model


def extract(record_id: str, data_dir: Path, window_samples: int, lead: int = 0):
    path = str(data_dir / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    signal = rec.p_signal[:, lead].astype(float)
    fs = rec.fs
    half = window_samples // 2
    xs, ys = [], []
    for idx, sym in zip(ann.sample, ann.symbol):
        if idx - half < 0 or idx + half > len(signal):
            continue
        if sym in NORMAL_SYMBOLS:
            y = 0
        elif sym in ANOMALY_SYMBOLS:
            y = 1
        else:
            continue
        raw = signal[idx - half:idx + half]
        if len(raw) != window_samples:
            continue
        pre = preprocess(raw, fs, apply_bandpass=True, apply_notch=False)
        xs.append(pre.astype(np.float32))
        ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default="data/mitbih")
    p.add_argument("--records", default=",".join(MITBIH_RECORDS[:10]),
                   help="comma-separated record ids")
    p.add_argument("--window", type=int, default=720)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    model = load_model(Path(args.model), ConvAutoencoder)

    Xs, Ys = [], []
    for rid in args.records.split(","):
        xs, ys = extract(rid.strip(), data_dir, args.window)
        if xs.size:
            Xs.append(xs); Ys.append(ys)
        print(f"  {rid}: {len(ys)} windows, {int(ys.sum())} anomalies")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)

    errors = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.from_numpy(X[i:i+256])
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.numpy())
    errors = np.concatenate(errors)

    prec, rec, thr = precision_recall_curve(y, errors)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    t_f1 = float(thr[best]) if best < len(thr) else float(thr[-1])

    normals = errors[y == 0]
    calib = {
        "method": "F1-optimal with real-time (per-window) preprocessing",
        "records_used": args.records.split(","),
        "n_windows": int(len(y)),
        "n_anomalies": int(y.sum()),
        "p99": t_f1,
        "threshold_f1_optimal": t_f1,
        "threshold_p99_normals": float(np.quantile(normals, 0.99)),
        "threshold_3std_normals": float(normals.mean() + 3 * normals.std()),
        "roc_auc": float(roc_auc_score(y, errors)),
        "f1_at_optimal": float(f1[best]),
        "precision_at_optimal": float(prec[best]),
        "recall_at_optimal": float(rec[best]),
    }
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
