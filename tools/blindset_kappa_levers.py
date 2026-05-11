"""Recompute Cohen's kappa on the 48-beat blind set under three improvement
levers, all on the SAME trained model — no retraining, no second clinician
ask.

Levers (cf. cardiologist-interview chapter §X.6):
    1. p99-percentile per-sample squared error (replaces mean MSE)
    2. mean MSE at a sensitivity-prioritised threshold (lower than F1-opt)
    3. three-class output {N, A, U} via a residual-magnitude noise gate

The cardiologist's 48 labels are fixed; we re-score the same 48 windows.

Output:
    - prints a comparison table
    - writes results/blindset_kappa_levers.json
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import wfdb

from config import (
    LATENT_DIM, MITBIH_DIR, WINDOW_SAMPLES, WINDOWS_PATH, LABELS_PATH,
    SEED, TRAIN_RATIO, VAL_RATIO,
)
from models import ConvAutoencoder
from preprocess import preprocess


CARDIO_LABELS = (
    "U,A,A,N,N,A,A,U,U,A,A,A,U,A,A,A,N,N,N,A,"
    "N,A,A,A,N,A,A,A,U,N,U,A,N,U,N,N,A,A,U,N,"
    "A,U,N,A,A,U,A,A"
).split(",")
assert len(CARDIO_LABELS) == 48

BLIND_KEY = Path("docs/clinical_proof/blind_answer_key.csv")
MODEL_PATH = Path("models/mitbih_autoencoder_best.pt")
OUT_PATH = Path("results/blindset_kappa_levers.json")

# Currently-deployed F1-optimal threshold (from per_symbol_test.json)
F1_THR_MEAN = 0.04339778
# F1-optimal threshold for p99, computed via sklearn precision_recall_curve
# in tools/evaluate_p99.py (NOT Youden-J). This is the threshold that would
# actually be deployed if we switched the score function.
F1_THR_P99 = 0.4562155604362488
# Lower threshold for mean MSE chosen so overall sensitivity rises to >=90%
# (from the per_symbol_multi_score script's analysis: at FPR=5% mean MSE
# sees 41.7% F sensitivity; the operating point we want is closer to the
# F-90% threshold of 0.0057, but that costs 40% FPR — so we pick a balanced
# middle: the threshold that gives 90% OVERALL sensitivity on the test
# split, computed below).


def _parse_source(s: str) -> tuple[str, int]:
    m = re.match(r"rec\s+(\S+),\s*sample\s+(\d+)", s)
    if not m:
        raise ValueError(f"unparseable source: {s}")
    return m.group(1), int(m.group(2))


def load_blind_beats():
    beats = []
    with BLIND_KEY.open() as f:
        for r in csv.DictReader(f):
            rid, sample = _parse_source(r["source"])
            beats.append({
                "beat_id": int(r["beat_id"]),
                "record": rid,
                "sample": sample,
                "gt_symbol": r["ground_truth_symbol"],
                "gt_class": r["ground_truth_class"],
                "stored_residual": float(r["model_residual"]),
                "stored_decision": r["model_decision"],
            })
    return beats


def extract_window(record: str, sample: int, lead: int = 0) -> np.ndarray:
    """Extract a 720-sample beat-centered window with the SAME preprocessing
    used by docs/clinical_proof/blind_answer_key.csv (clinical_proof.py:67):
    bandpass over the whole record then cut a 720-sample window centered on
    the annotated R-peak. The residuals match the stored ones to ~6 decimals
    when this preprocessing is identical."""
    rec = wfdb.rdrecord(str(MITBIH_DIR / record))
    raw = rec.p_signal[:, lead].astype(float)
    fs = int(rec.fs)
    sig = preprocess(raw, fs, apply_bandpass=True, apply_notch=False)
    half = WINDOW_SAMPLES // 2
    if sample - half < 0 or sample + half > len(sig):
        raise ValueError(f"window out of range for {record}@{sample}")
    return sig[sample - half:sample + half].astype(np.float32)


def score_window(model, x: np.ndarray) -> dict:
    """Run the model on a single 720-sample window and return all the
    per-window summaries we'll need for the three levers."""
    with torch.no_grad():
        xt = torch.from_numpy(x[None, :])
        rec = model(xt).numpy()[0]
    err = (x - rec) ** 2
    return {
        "mean_mse": float(err.mean()),
        "max_err":  float(err.max()),
        "p99":      float(np.percentile(err, 99)),
        "signal_std": float(x.std()),
        "signal_range": float(x.max() - x.min()),
        "deriv_std": float(np.diff(x).std()),  # noise proxy
    }


def kappa(a: list[str], b: list[str]) -> tuple[float, float, float]:
    """Cohen's kappa, returns (kappa, observed_agreement, expected_agreement)."""
    assert len(a) == len(b)
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), po, pe


def evaluate_lever(name: str, model_labels: list[str], cardio: list[str],
                   has_u: bool) -> dict:
    """Compare a model labelling (binary or ternary) against the cardiologist
    labels under three U-handling rules."""
    out = {"name": name}

    # Drop-U
    pairs = [(c, m) for c, m in zip(cardio, model_labels) if c != "U"]
    ca, mb = (list(z) for z in zip(*pairs)) if pairs else ([], [])
    k, po, pe = kappa(ca, mb)
    out["dropU"] = {"n": len(pairs), "kappa": k, "po": po, "pe": pe}

    # U -> A
    ca2 = ["A" if c == "U" else c for c in cardio]
    k, po, pe = kappa(ca2, model_labels)
    out["U_to_A"] = {"n": 48, "kappa": k, "po": po, "pe": pe}

    # U -> N
    ca3 = ["N" if c == "U" else c for c in cardio]
    k, po, pe = kappa(ca3, model_labels)
    out["U_to_N"] = {"n": 48, "kappa": k, "po": po, "pe": pe}

    # Confusion matrix (drop-U for binary, full for ternary)
    if has_u:
        out["confusion_full"] = {
            cl: {ml: sum(1 for c, m in zip(cardio, model_labels)
                         if c == cl and m == ml)
                 for ml in ["N", "A", "U"]}
            for cl in ["N", "A", "U"]
        }
    else:
        out["confusion_dropU"] = {
            cl: {ml: sum(1 for c, m in zip(ca, mb) if c == cl and m == ml)
                 for ml in ["N", "A"]}
            for cl in ["N", "A"]
        }

    return out


def find_threshold_at_target_sens(score_fn, target_sens: float = 0.90):
    """Recompute the threshold for an arbitrary score function on the test
    split such that overall sensitivity = target_sens."""
    print(f"  finding threshold at overall sensitivity = {target_sens:.0%}...")
    X = np.load(WINDOWS_PATH).astype(np.float32)
    y = np.load(LABELS_PATH)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_test, y_test = X[test_start:], y[test_start:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    last_channels = int(state["encoder.9.weight"].shape[0])
    model = ConvAutoencoder(ckpt["input_dim"], ckpt.get("latent_dim", LATENT_DIM),
                            last_channels=last_channels)
    model.load_state_dict(state); model.to(device).eval()

    scores = np.empty(len(X_test), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            b = torch.from_numpy(X_test[i:i+512]).to(device)
            r = model(b).cpu().numpy()
            scores[i:i+512] = score_fn(X_test[i:i+512], r)

    anom_scores = np.sort(scores[y_test == 1])
    k = int(np.ceil((1.0 - target_sens) * len(anom_scores)))
    return float(anom_scores[max(0, k - 1)])


def main() -> None:
    beats = load_blind_beats()
    assert len(beats) == 48

    print("Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    last_channels = int(state["encoder.9.weight"].shape[0])
    model = ConvAutoencoder(ckpt["input_dim"], ckpt.get("latent_dim", LATENT_DIM),
                            last_channels=last_channels)
    model.load_state_dict(state); model.eval()

    print("Re-scoring 48 blind beats from MIT-BIH source records...")
    for b in beats:
        x = extract_window(b["record"], b["sample"])
        b.update(score_window(model, x))
    # Sanity check: stored mean_mse should match recomputed mean_mse closely.
    diffs = [abs(b["mean_mse"] - b["stored_residual"]) for b in beats]
    print(f"  recomputed-vs-stored mean_mse: max diff = {max(diffs):.6f}, "
          f"mean diff = {np.mean(diffs):.6f}")

    # Lever 2: find a sensitivity-prioritised threshold for mean MSE
    print("\nCalibrating sensitivity-prioritised threshold for mean MSE...")
    thr_mean_sens = find_threshold_at_target_sens(
        lambda x, r: np.mean((x - r) ** 2, axis=1), target_sens=0.95
    )
    print(f"  -> mean-MSE threshold @ 95% test sensitivity = {thr_mean_sens:.4f}")

    # ------------------------------------------------------------------
    # Build per-lever model labellings on the 48-set
    # ------------------------------------------------------------------

    # Baseline: current deployed system (mean MSE @ F1-optimal)
    base_labels = ["A" if b["mean_mse"] > F1_THR_MEAN else "N" for b in beats]

    # Lever 1: p99 reduction at its own F1-optimal threshold
    l1_labels = ["A" if b["p99"] > F1_THR_P99 else "N" for b in beats]

    # Lever 2: mean MSE at the lower sensitivity-prioritised threshold
    l2_labels = ["A" if b["mean_mse"] > thr_mean_sens else "N" for b in beats]

    # Lever 5: p99 + sensitivity-tuned threshold (best of both)
    print("Calibrating sensitivity-prioritised threshold for p99...")
    thr_p99_sens = find_threshold_at_target_sens(
        lambda x, r: np.percentile((x - r) ** 2, 99, axis=1),
        target_sens=0.95,
    )
    print(f"  -> p99 threshold @ 95% test sensitivity = {thr_p99_sens:.4f}")
    l5_labels = ["A" if b["p99"] > thr_p99_sens else "N" for b in beats]

    # Lever 3: three-class output with a noise gate, on top of mean-MSE F1-opt.
    U_RES_GATE = 10 * F1_THR_MEAN     # ~ 0.43 — order of magnitude over normal
    U_DERIV_GATE = 0.40                # empirical: noisy windows > 0.4
    l3_labels = []
    for b in beats:
        if b["mean_mse"] > U_RES_GATE and b["deriv_std"] > U_DERIV_GATE:
            l3_labels.append("U")
        elif b["mean_mse"] > F1_THR_MEAN:
            l3_labels.append("A")
        else:
            l3_labels.append("N")

    # Lever 4: combine lever 1 (p99 score) with lever 3 (noise gate).
    # The U gate uses the same noise heuristic; the binary decision
    # underneath is the more sensitive p99-at-F1-opt rule.
    l4_labels = []
    for b in beats:
        if b["mean_mse"] > U_RES_GATE and b["deriv_std"] > U_DERIV_GATE:
            l4_labels.append("U")
        elif b["p99"] > F1_THR_P99:
            l4_labels.append("A")
        else:
            l4_labels.append("N")

    # ------------------------------------------------------------------
    # Compute kappa under each lever
    # ------------------------------------------------------------------
    results = {
        "baseline":            evaluate_lever("baseline_meanMSE_F1opt",
                                              base_labels, CARDIO_LABELS, has_u=False),
        "lever1_p99":          evaluate_lever("p99_F1opt",
                                              l1_labels, CARDIO_LABELS, has_u=False),
        "lever2_mean_sens":    evaluate_lever(f"meanMSE_sensTuned_thr={thr_mean_sens:.4f}",
                                              l2_labels, CARDIO_LABELS, has_u=False),
        "lever3_three_class":  evaluate_lever("meanMSE_with_noise_gate_three_class",
                                              l3_labels, CARDIO_LABELS, has_u=True),
        "lever4_combined":     evaluate_lever("p99_with_noise_gate_three_class",
                                              l4_labels, CARDIO_LABELS, has_u=True),
        "lever5_p99_sens":     evaluate_lever(f"p99_sensTuned_thr={thr_p99_sens:.4f}",
                                              l5_labels, CARDIO_LABELS, has_u=False),
        "_meta": {
            "F1_threshold_mean_mse": F1_THR_MEAN,
            "F1_threshold_p99": F1_THR_P99,
            "sensitivity_tuned_threshold_mean_mse": thr_mean_sens,
            "three_class_residual_gate": U_RES_GATE,
            "three_class_deriv_gate": U_DERIV_GATE,
        },
        "per_beat": [
            {**{k: b[k] for k in ["beat_id", "record", "sample",
                                  "gt_symbol", "gt_class",
                                  "mean_mse", "p99", "max_err",
                                  "signal_std", "deriv_std"]},
             "cardio": CARDIO_LABELS[i],
             "baseline": base_labels[i],
             "lever1_p99": l1_labels[i],
             "lever2_meanMSE_sens": l2_labels[i],
             "lever3_three_class": l3_labels[i],
             "lever4_combined": l4_labels[i],
             "lever5_p99_sens": l5_labels[i]}
            for i, b in enumerate(beats)
        ],
    }

    print(f"\n{'lever':<35} {'drop-U κ':>10} {'U→A κ':>10} {'U→N κ':>10}")
    print("-" * 70)
    for k in ["baseline", "lever1_p99", "lever2_mean_sens",
              "lever3_three_class", "lever4_combined", "lever5_p99_sens"]:
        r = results[k]
        print(f"{r['name']:<35} "
              f"{r['dropU']['kappa']:>10.4f} "
              f"{r['U_to_A']['kappa']:>10.4f} "
              f"{r['U_to_N']['kappa']:>10.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
