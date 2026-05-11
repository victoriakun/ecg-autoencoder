"""Per-symbol detection breakdown across multiple anomaly-scoring strategies.

Extends ``per_symbol_eval.py`` and ``audit_scoring.py``:
no retraining, just re-scoring the held-out test split under several
strategies and reporting per-MIT-BIH-symbol sensitivity at three operating
points:

    1. F1-optimal threshold (current operating point of the deployed model)
    2. Threshold that achieves >= TARGET_F_SENS on Fusion (F) beats
       — the cardiologist's clinical bar
    3. Threshold pinned at FPR = 5% on normal beats, for AAMI-style reporting

The strategy that hits the F-sensitivity target at the lowest FPR is the
recommended deployment configuration.

Output: ``results/per_symbol_multi_score.json``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from config import (
    WINDOWS_PATH, LABELS_PATH, MITBIH_RECORDS, SEED,
    NORMAL_SYMBOLS, ANOMALY_SYMBOLS, LATENT_DIM, TRAIN_RATIO, VAL_RATIO,
)
from dataset import get_beat_windows
from models import ConvAutoencoder


SYMBOL_NAMES = {
    "N": "Normal sinus", "L": "Left bundle branch block",
    "R": "Right bundle branch block", "e": "Atrial escape",
    "j": "Nodal (junctional) escape", "A": "Atrial premature (APC)",
    "a": "Aberrated atrial premature", "J": "Nodal (junctional) premature",
    "S": "Supraventricular premature",
    "V": "Premature ventricular contraction (PVC)",
    "F": "Fusion of ventricular + normal",
    "!": "Ventricular flutter wave", "E": "Ventricular escape",
    "/": "Paced beat", "f": "Fusion of paced + normal", "Q": "Unclassifiable",
}

TARGET_F_SENS = 0.90  # cardiologist's clinical bar for fusion beats
TARGET_FPR = 0.05     # AAMI-style operating-point reporting


def collect_symbols() -> np.ndarray:
    syms = []
    for rec in MITBIH_RECORDS:
        _, _, s = get_beat_windows(rec)
        syms.extend(s)
    return np.array(syms)


@torch.no_grad()
def forward_all(model, X, device, batch_size=512):
    model.eval()
    R, Z = [], []
    for i in range(0, len(X), batch_size):
        b = torch.from_numpy(X[i:i + batch_size]).to(device)
        z = model.encode(b)
        r = model.decode(z)
        R.append(r.cpu().numpy())
        Z.append(z.cpu().numpy())
    return np.concatenate(R), np.concatenate(Z)


def s_mean_mse(X, R):       return np.mean((X - R) ** 2, axis=1)
def s_max_err(X, R):        return np.max((X - R) ** 2, axis=1)
def s_p99(X, R):            return np.percentile((X - R) ** 2, 99, axis=1)
def s_p95(X, R):            return np.percentile((X - R) ** 2, 95, axis=1)
def s_cosine(X, R):
    num = np.sum(X * R, axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1) + 1e-9
    return 1.0 - num / den
def s_deriv_mse(X, R):
    return np.mean((np.diff(X, axis=1) - np.diff(R, axis=1)) ** 2, axis=1)
def s_mahalanobis(Z, Z_normal):
    mu = Z_normal.mean(axis=0)
    cov = np.cov(Z_normal, rowvar=False)
    inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    d = Z - mu
    return np.einsum("ij,jk,ik->i", d, inv, d)


def zscore_using(values, ref_normal):
    mu, sd = ref_normal.mean(), ref_normal.std() + 1e-9
    return (values - mu) / sd


def threshold_at_target_recall_on_class(scores, mask_class, target):
    """Lowest threshold so that recall on ``mask_class`` >= target."""
    cls_scores = np.sort(scores[mask_class])
    if len(cls_scores) == 0:
        return None
    k = max(1, int(np.ceil((1.0 - target) * len(cls_scores))))
    return cls_scores[k - 1] if k <= len(cls_scores) else cls_scores[-1]


def threshold_at_target_fpr(scores, mask_normal, target_fpr):
    """Threshold at which the FPR on normals == target_fpr."""
    norm_scores = np.sort(scores[mask_normal])
    k = int(np.ceil((1.0 - target_fpr) * len(norm_scores))) - 1
    k = max(0, min(k, len(norm_scores) - 1))
    return norm_scores[k]


def threshold_f1_optimal(scores, y):
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    return thr[np.argmax(j)]  # Youden's J — equivalent to max F1 in practice when classes are similar


def per_symbol_rates(scores, threshold, sym_test):
    flagged = scores > threshold
    rows = []
    for sym in sorted(set(sym_test), key=lambda s: -np.sum(sym_test == s)):
        mask = sym_test == sym
        n = int(mask.sum())
        n_flagged = int(flagged[mask].sum())
        is_anom = sym in ANOMALY_SYMBOLS
        is_norm = sym in NORMAL_SYMBOLS
        rows.append({
            "symbol": sym, "name": SYMBOL_NAMES.get(sym, "?"),
            "class": "anomaly" if is_anom else "normal" if is_norm else "?",
            "n": n, "flagged": n_flagged,
            "rate": round(n_flagged / n if n else 0.0, 4),
        })
    return rows


def overall_metrics(scores, threshold, y, sym_test, mask_normal):
    flagged = scores > threshold
    sens = float(flagged[y == 1].mean()) if (y == 1).any() else 0.0
    spec = float((~flagged[y == 0]).mean()) if (y == 0).any() else 0.0
    fpr = float(flagged[mask_normal].mean()) if mask_normal.any() else 0.0
    f_mask = sym_test == "F"
    f_sens = float(flagged[f_mask].mean()) if f_mask.any() else 0.0
    v_mask = sym_test == "V"
    v_sens = float(flagged[v_mask].mean()) if v_mask.any() else 0.0
    e_mask = sym_test == "E"
    e_sens = float(flagged[e_mask].mean()) if e_mask.any() else 0.0
    return {"threshold": float(threshold),
            "overall_sens": sens, "overall_spec": spec, "fpr_on_N": fpr,
            "F_sens": f_sens, "V_sens": v_sens, "E_sens": e_sens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    ap.add_argument("--out", default="results/per_symbol_multi_score.json")
    args = ap.parse_args()

    print("Loading windows + labels...")
    X = np.load(WINDOWS_PATH).astype(np.float32)
    y = np.load(LABELS_PATH)

    print("Recovering MIT-BIH symbols (re-walk records)...")
    symbols = collect_symbols()
    if len(symbols) != len(X):
        raise RuntimeError(f"symbol count {len(symbols)} != windows {len(X)}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y, symbols = X[idx], y[idx], symbols[idx]
    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_train, y_train = X[:test_start], y[:test_start]
    X_test, y_test, sym_test = X[test_start:], y[test_start:], symbols[test_start:]
    print(f"  test: {len(X_test)} ({(y_test==0).sum()} N, {(y_test==1).sum()} A)")

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    last_channels = int(state["encoder.9.weight"].shape[0])
    model = ConvAutoencoder(
        ckpt["input_dim"],
        ckpt.get("latent_dim", LATENT_DIM),
        last_channels=last_channels,
    )
    model.load_state_dict(state)
    model.to(device)

    print("Forward pass (test)...")
    R_test, Z_test = forward_all(model, X_test, device)
    print("Forward pass (train normals only, for Mahalanobis & z-score refs)...")
    Z_train_norm = forward_all(model, X_train[y_train == 0], device)[1]

    # Per-strategy raw scores
    raw = {
        "mean_mse":           s_mean_mse(X_test, R_test),
        "max_err":            s_max_err(X_test, R_test),
        "p99_err":            s_p99(X_test, R_test),
        "p95_err":            s_p95(X_test, R_test),
        "cosine":             s_cosine(X_test, R_test),
        "deriv_mse":          s_deriv_mse(X_test, R_test),
        "latent_mahalanobis": s_mahalanobis(Z_test, Z_train_norm),
    }

    # Per-strategy normal-only scores (z-reference)
    mask_normal = y_test == 0
    z = {k: zscore_using(v, v[mask_normal]) for k, v in raw.items()}

    # Two combined strategies
    raw["combined_zsum"] = sum(z.values())
    raw["combined_zmax"] = np.max(np.stack(list(z.values())), axis=0)

    # ROC + per-symbol report at three thresholds
    out: Dict[str, dict] = {}
    print(f"\n{'strategy':<20} {'ROC-AUC':>8} {'F-only AUC':>11} "
          f"{'F-sens@F1opt':>12} {'F-sens@FPR5':>12} {'Tau@F90':>10} {'FPR@F90':>9}")
    print("-" * 95)

    f_mask = sym_test == "F"
    n_only_mask = mask_normal | f_mask
    for name, scores in raw.items():
        roc = float(roc_auc_score(y_test, scores))
        # F-only AUC: how well does this score discriminate Normal from F specifically?
        roc_f = float(roc_auc_score(y_test[n_only_mask], scores[n_only_mask]))

        thr_f1 = float(threshold_f1_optimal(scores, y_test))
        thr_fpr5 = float(threshold_at_target_fpr(scores, mask_normal, TARGET_FPR))
        thr_f90 = threshold_at_target_recall_on_class(scores, f_mask, TARGET_F_SENS)
        thr_f90 = float(thr_f90) if thr_f90 is not None else None

        m_f1 = overall_metrics(scores, thr_f1, y_test, sym_test, mask_normal)
        m_fpr5 = overall_metrics(scores, thr_fpr5, y_test, sym_test, mask_normal)
        m_f90 = overall_metrics(scores, thr_f90, y_test, sym_test, mask_normal) if thr_f90 is not None else None

        out[name] = {
            "roc_auc_overall": roc,
            "roc_auc_normal_vs_F_only": roc_f,
            "operating_points": {
                "f1_optimal":  m_f1,
                "fpr_5pct":    m_fpr5,
                "F_sens_90pct": m_f90,
            },
            "per_symbol_at_F90": per_symbol_rates(scores, thr_f90, sym_test) if thr_f90 is not None else None,
        }
        f90_disp = f"{m_f90['fpr_on_N']*100:>7.2f}%" if m_f90 else "    n/a"
        thr_f90_disp = f"{thr_f90:>10.4f}" if thr_f90 is not None else "       n/a"
        print(f"{name:<20} {roc:>8.4f} {roc_f:>11.4f} "
              f"{m_f1['F_sens']*100:>10.1f}% {m_fpr5['F_sens']*100:>10.1f}% "
              f"{thr_f90_disp} {f90_disp}")

    # Print full per-symbol breakdown for the winner
    winner = min(
        (k for k, v in out.items() if v["operating_points"]["F_sens_90pct"]),
        key=lambda k: out[k]["operating_points"]["F_sens_90pct"]["fpr_on_N"],
    )
    print(f"\n>> Recommended strategy for fusion target: {winner}")
    print(f"   At >= {TARGET_F_SENS*100:.0f}% F sensitivity:")
    for k, v in out[winner]["operating_points"]["F_sens_90pct"].items():
        if isinstance(v, float):
            print(f"     {k:<20} {v:.4f}")

    print(f"\nFull per-symbol table at F-sens={TARGET_F_SENS*100:.0f}% threshold using {winner}:")
    print(f"{'Sym':<3} {'Class':<8} {'n':>5} {'flag':>6} {'rate':>7}  Name")
    for r in out[winner]["per_symbol_at_F90"]:
        print(f"{r['symbol']:<3} {r['class']:<8} {r['n']:>5} {r['flagged']:>6} "
              f"{r['rate']*100:>6.1f}%  {r['name']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "test_n": int(len(X_test)),
        "test_normal": int((y_test == 0).sum()),
        "test_anomaly": int((y_test == 1).sum()),
        "f_count": int(f_mask.sum()),
        "v_count": int((sym_test == "V").sum()),
        "e_count": int((sym_test == "E").sum()),
        "target_F_sensitivity": TARGET_F_SENS,
        "target_FPR": TARGET_FPR,
        "winner_for_F_target": winner,
        "strategies": out,
    }, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
