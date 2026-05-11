"""Full evaluation of the deployed model under the p99-percentile score.

No retraining. Same model weights, same test split as the existing
per_symbol_eval.py / audit_scoring.py — only the per-window score
reduction changes from mean(err) to percentile(err, 99).

Reports a side-by-side comparison of:
    ROC-AUC, PR-AUC, F1-optimal threshold, sensitivity, specificity,
    precision, F1, balanced accuracy, plus a per-symbol breakdown.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve,
)

from config import (
    WINDOWS_PATH, LABELS_PATH, MITBIH_RECORDS, SEED,
    NORMAL_SYMBOLS, ANOMALY_SYMBOLS, LATENT_DIM, TRAIN_RATIO, VAL_RATIO,
)
from dataset import get_beat_windows
from models import ConvAutoencoder

SYMBOL_NAMES = {
    "N": "Normal sinus", "L": "LBBB", "R": "RBBB", "e": "Atrial escape",
    "j": "Junctional escape", "A": "APC", "a": "Aberrated APC",
    "J": "Junctional premature", "S": "SVE", "V": "PVC",
    "F": "Fusion (V+N)", "!": "VFL wave", "E": "Vent. escape",
    "/": "Paced", "f": "Fusion (paced+N)", "Q": "Unclassifiable",
}


def collect_symbols():
    syms = []
    for rec in MITBIH_RECORDS:
        _, _, s = get_beat_windows(rec)
        syms.extend(s)
    return np.array(syms)


@torch.no_grad()
def reconstruct(model, X, device, bs=512):
    model.eval()
    out = np.empty_like(X)
    for i in range(0, len(X), bs):
        b = torch.from_numpy(X[i:i+bs]).to(device)
        out[i:i+bs] = model(b).cpu().numpy()
    return out


def metrics_at_threshold(scores, y, thr):
    pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "sensitivity_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity":        float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision":          float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "f1":                 float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy":  float(0.5 * (tp / (tp + fn) + tn / (tn + fp))),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def f1_optimal_threshold(scores, y):
    p, r, t = precision_recall_curve(y, scores)
    f = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f[:-1])])  # last point has no threshold


def per_symbol(scores, thr, sym):
    flagged = scores > thr
    rows = []
    for s in sorted(set(sym), key=lambda v: -np.sum(sym == v)):
        m = sym == s
        n = int(m.sum()); k = int(flagged[m].sum())
        cls = ("anomaly" if s in ANOMALY_SYMBOLS
               else "normal" if s in NORMAL_SYMBOLS else "?")
        rows.append({"symbol": s, "name": SYMBOL_NAMES.get(s, "?"),
                     "class": cls, "n": n, "flagged": k,
                     "rate": round(k / n, 4) if n else 0.0})
    return rows


def main():
    X = np.load(WINDOWS_PATH).astype(np.float32)
    y = np.load(LABELS_PATH)
    syms = collect_symbols()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y, syms = X[idx], y[idx], syms[idx]
    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_test, y_test, sym_test = X[test_start:], y[test_start:], syms[test_start:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("models/mitbih_autoencoder_best.pt",
                      map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    last_channels = int(state["encoder.9.weight"].shape[0])
    model = ConvAutoencoder(ckpt["input_dim"], ckpt.get("latent_dim", LATENT_DIM),
                            last_channels=last_channels)
    model.load_state_dict(state); model.to(device).eval()

    R_test = reconstruct(model, X_test, device)
    err = (X_test - R_test) ** 2

    score_baseline = err.mean(axis=1)              # mean MSE — current
    score_p99      = np.percentile(err, 99, axis=1)  # Lever 1

    out = {}
    for name, scores in [("baseline_mean_mse", score_baseline),
                         ("lever1_p99",       score_p99)]:
        roc_auc = float(roc_auc_score(y_test, scores))
        pr_auc  = float(average_precision_score(y_test, scores))
        thr     = f1_optimal_threshold(scores, y_test)
        m       = metrics_at_threshold(scores, y_test, thr)
        # Operating point at fixed 5% FPR (AAMI-style)
        fpr_curve, tpr_curve, t_curve = roc_curve(y_test, scores)
        thr_5pct = float(t_curve[np.argmin(np.abs(fpr_curve - 0.05))])
        m_5 = metrics_at_threshold(scores, y_test, thr_5pct)
        out[name] = {
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "f1_optimal": m, "fpr_5pct": m_5,
            "per_symbol": per_symbol(scores, thr, sym_test),
        }

    # Side-by-side
    print(f"{'metric':<22}{'baseline':>14}{'lever1 p99':>14}{'Δ':>10}")
    print("-" * 62)
    base, l1 = out["baseline_mean_mse"], out["lever1_p99"]
    for k in ["roc_auc", "pr_auc"]:
        print(f"{k:<22}{base[k]:>14.4f}{l1[k]:>14.4f}{l1[k]-base[k]:>+10.4f}")
    print()
    print(f"{'@ F1-optimal threshold':<22}")
    for k in ["threshold", "sensitivity_recall", "specificity",
              "precision", "f1", "balanced_accuracy", "fpr"]:
        b, a = base["f1_optimal"][k], l1["f1_optimal"][k]
        print(f"  {k:<20}{b:>14.4f}{a:>14.4f}{a-b:>+10.4f}")
    bo = base["f1_optimal"]; lo = l1["f1_optimal"]
    base_cm = f"{bo['TP']}/{bo['FP']}/{bo['TN']}/{bo['FN']}"
    l1_cm   = f"{lo['TP']}/{lo['FP']}/{lo['TN']}/{lo['FN']}"
    print(f"  {'TP/FP/TN/FN':<20}{base_cm:>14}{l1_cm:>14}")
    print()
    print(f"{'@ FPR=5% threshold':<22}")
    for k in ["threshold", "sensitivity_recall", "specificity",
              "precision", "f1"]:
        b, a = base["fpr_5pct"][k], l1["fpr_5pct"][k]
        print(f"  {k:<20}{b:>14.4f}{a:>14.4f}{a-b:>+10.4f}")

    print(f"\n{'Per-symbol sensitivity at F1-optimal threshold':}")
    print(f"{'sym':<4}{'class':<9}{'n':>5}{'baseline':>12}{'p99':>10}{'Δ':>8}  name")
    base_rows = {r["symbol"]: r for r in base["per_symbol"]}
    for r in l1["per_symbol"]:
        b = base_rows[r["symbol"]]
        delta = (r["rate"] - b["rate"]) * 100
        print(f"{r['symbol']:<4}{r['class']:<9}{r['n']:>5}"
              f"{b['rate']*100:>11.1f}%{r['rate']*100:>9.1f}%"
              f"{delta:>+7.1f}%  {r['name']}")

    Path("results").mkdir(exist_ok=True)
    Path("results/evaluate_p99.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> results/evaluate_p99.json")


if __name__ == "__main__":
    main()
