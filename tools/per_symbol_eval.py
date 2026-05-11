"""Per-symbol detection breakdown on the held-out MIT-BIH test split.

Reproduces the 70/15/15 shuffle from train_mitbih.py with seed=42, then
reports for each annotation symbol:
  - count in the test split
  - count caught (residual > threshold)
  - detection rate (sensitivity for anomalies, false-alarm rate for normals)

Threshold defaults to the model's stored F1-optimal value (0.0434).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import (
    WINDOWS_PATH, LABELS_PATH, MITBIH_RECORDS, SEED,
    NORMAL_SYMBOLS, ANOMALY_SYMBOLS,
)
from dataset import get_beat_windows
from models import ConvAutoencoder
from realtime.inference import load_model


SYMBOL_NAMES = {
    "N": "Normal sinus",
    "L": "Left bundle branch block",
    "R": "Right bundle branch block",
    "e": "Atrial escape",
    "j": "Nodal (junctional) escape",
    "A": "Atrial premature (APC)",
    "a": "Aberrated atrial premature",
    "J": "Nodal (junctional) premature",
    "S": "Supraventricular premature",
    "V": "Premature ventricular contraction (PVC)",
    "F": "Fusion of ventricular + normal",
    "!": "Ventricular flutter wave",
    "E": "Ventricular escape",
    "/": "Paced beat",
    "f": "Fusion of paced + normal",
    "Q": "Unclassifiable",
}


def collect_symbols() -> list[str]:
    """Re-walk the records in the same order as build_dataset() to recover
    the symbol for every saved window."""
    symbols: list[str] = []
    for rec in MITBIH_RECORDS:
        _, _, syms = get_beat_windows(rec)
        symbols.extend(syms)
    return symbols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/mitbih_autoencoder_best.pt")
    p.add_argument("--threshold", type=float, default=0.04339778050780296,
                   help="Stored F1-optimal threshold from the checkpoint.")
    p.add_argument("--out", default="results/per_symbol_test.json")
    args = p.parse_args()

    print("Loading saved windows + labels...")
    X = np.load(WINDOWS_PATH).astype(np.float32)
    y = np.load(LABELS_PATH)
    print(f"  loaded {len(X)} windows")

    print("Recovering symbols by re-walking records...")
    symbols = collect_symbols()
    if len(symbols) != len(X):
        raise RuntimeError(
            f"symbol count {len(symbols)} != window count {len(X)} — "
            "record set or preprocessing differs.")
    symbols = np.array(symbols)

    # Same shuffle and split as train_mitbih.py
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y, symbols = X[idx], y[idx], symbols[idx]
    val_end = int(0.85 * len(X))
    X_test, y_test, sym_test = X[val_end:], y[val_end:], symbols[val_end:]
    print(f"  test split: {len(X_test)} windows "
          f"({(y_test == 0).sum()} normal, {(y_test == 1).sum()} anomaly)")

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model), ConvAutoencoder).to(device)
    model.eval()

    print("Computing reconstruction errors on test split...")
    errors = np.empty(len(X_test), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            batch = torch.from_numpy(X_test[i:i+512]).to(device)
            recon = model(batch)
            errors[i:i+512] = torch.mean((recon - batch) ** 2, dim=1).cpu().numpy()

    flagged = errors > args.threshold
    overall_sens = flagged[y_test == 1].mean() if (y_test == 1).any() else 0.0
    overall_spec = (~flagged[y_test == 0]).mean() if (y_test == 0).any() else 0.0
    print(f"\nOverall on test split (threshold={args.threshold:.4f}):")
    print(f"  sensitivity = {overall_sens:.3f}")
    print(f"  specificity = {overall_spec:.3f}")

    # Per-symbol breakdown
    rows = []
    for sym in sorted(set(sym_test), key=lambda s: -np.sum(sym_test == s)):
        mask = sym_test == sym
        n = int(mask.sum())
        n_flagged = int(flagged[mask].sum())
        is_anomaly = sym in ANOMALY_SYMBOLS
        is_normal = sym in NORMAL_SYMBOLS
        rate = n_flagged / n if n else 0.0
        rows.append({
            "symbol": sym,
            "name": SYMBOL_NAMES.get(sym, "?"),
            "class": "anomaly" if is_anomaly else "normal" if is_normal else "?",
            "n_test": n,
            "n_flagged": n_flagged,
            "rate": round(rate, 4),
        })

    print(f"\n{'Sym':<3} {'Class':<8} {'n':>6} {'flagged':>8} {'rate':>7}  Name")
    print("-" * 80)
    for r in rows:
        kind_label = "sens" if r["class"] == "anomaly" else "FPR"
        print(f"{r['symbol']:<3} {r['class']:<8} {r['n_test']:>6} "
              f"{r['n_flagged']:>8} {r['rate']*100:>6.1f}%  "
              f"({kind_label}) {r['name']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "threshold": args.threshold,
        "test_n": len(X_test),
        "test_normal": int((y_test == 0).sum()),
        "test_anomaly": int((y_test == 1).sum()),
        "overall_sensitivity": float(overall_sens),
        "overall_specificity": float(overall_spec),
        "per_symbol": rows,
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
