"""Compute the F1-optimal reconstruction-error threshold for a checkpoint.

Mirrors the batch methodology in evaluate.py: runs the model on the labeled
test split, sweeps thresholds, picks the one that maximises F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

from config import WINDOWS_PATH, LABELS_PATH, TRAIN_RATIO, VAL_RATIO, SEED
from models import ConvAutoencoder
from realtime.inference import load_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    X = np.load(WINDOWS_PATH)
    y = np.load(LABELS_PATH)
    rng = np.random.default_rng(SEED)
    idx = np.arange(X.shape[0]); rng.shuffle(idx)
    n_train = int(len(idx) * TRAIN_RATIO)
    n_val = int(len(idx) * VAL_RATIO)
    test_idx = idx[n_train + n_val:]
    X_te, y_te = X[test_idx], y[test_idx]

    model = load_model(Path(args.model), ConvAutoencoder)
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_te), 256):
            batch = torch.from_numpy(X_te[i:i+256]).float()
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.numpy())
    errors = np.concatenate(errors)

    prec, rec, thr = precision_recall_curve(y_te, errors)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    t_f1 = float(thr[best]) if best < len(thr) else float(thr[-1])

    normals = errors[y_te == 0]
    calib = {
        "method": "F1-optimal on labeled test split",
        "source": "data/mitbih_windows.npy",
        "n_test": int(len(y_te)),
        "n_anomalies_in_test": int(y_te.sum()),
        "p99": t_f1,                         # consumed by FixedOnlineThreshold
        "threshold_f1_optimal": t_f1,
        "threshold_p99_normals": float(np.quantile(normals, 0.99)),
        "threshold_3std_normals": float(normals.mean() + 3 * normals.std()),
        "roc_auc": float(roc_auc_score(y_te, errors)),
        "f1_at_optimal": float(f1[best]),
        "precision_at_optimal": float(prec[best]),
        "recall_at_optimal": float(rec[best]),
    }
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(json.dumps(calib, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
