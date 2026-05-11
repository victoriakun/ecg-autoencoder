"""Generate the three NEW figures the rewritten Chapter 6 needs:

  Figure 6.6  Per-class detection rates at the F1-optimal threshold
  Figure 6.7  Cohen's kappa under three operating points
  Figure 6.8  Score-function comparison (mean MSE vs p99) on priority classes

All figures are saved to evaluation_plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path("evaluation_plots")
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS = Path("results")


def fig_per_class_rates() -> None:
    """Bar chart of per-AAMI-symbol detection rate at the deployed threshold."""
    d = json.loads((RESULTS / "per_symbol_test.json").read_text())
    rows = sorted(d["per_symbol"], key=lambda r: -r["rate"])
    labels = [r["symbol"] for r in rows]
    rates  = [r["rate"] * 100 for r in rows]
    cls    = [r["class"]      for r in rows]
    names  = [r["name"]       for r in rows]
    colors = ["#1f77b4" if c == "anomaly" else "#888" for c in cls]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.6)
    for b, r, n in zip(bars, rates, names):
        ax.text(b.get_x() + b.get_width()/2, r + 1.5,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)", fontsize=10)
    ax.set_xlabel("AAMI annotation symbol", fontsize=10)
    ax.set_title(
        f"Per-symbol detection rate at the F1-optimal threshold "
        f"(τ = {d['threshold']:.4f})", fontsize=11, fontweight="bold",
    )
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1f77b4", label="anomaly classes (sensitivity)"),
        Patch(color="#888",    label="normal classes (false-positive rate)"),
    ], loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "7_per_class_detection.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_kappa_levers() -> None:
    """Cohen's kappa under three U-treatments and two operating points."""
    d = json.loads((RESULTS / "blindset_kappa_levers.json").read_text())
    base = d["baseline"]
    best = d["lever5_p99_sens"]  # the recommended deployment point

    treatments = ["drop-U", "U → A", "U → N"]
    base_vals = [base["dropU"]["kappa"], base["U_to_A"]["kappa"],
                 base["U_to_N"]["kappa"]]
    best_vals = [best["dropU"]["kappa"], best["U_to_A"]["kappa"],
                 best["U_to_N"]["kappa"]]

    x = np.arange(len(treatments))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    # Landis-Koch interpretation bands
    for ymin, ymax, label, color in [
        (0.00, 0.20, "slight",     "#fff5f0"),
        (0.20, 0.40, "fair",       "#fee0d2"),
        (0.40, 0.60, "moderate",   "#fcbba1"),
        (0.60, 0.80, "substantial","#fc9272"),
        (0.80, 1.00, "almost perfect", "#fb6a4a"),
    ]:
        ax.axhspan(ymin, ymax, color=color, alpha=0.5, zorder=0)
        ax.text(2.55, (ymin + ymax) / 2, label, va="center", ha="left",
                fontsize=8, color="#444")

    bars1 = ax.bar(x - w/2, base_vals, w, label="baseline (mean MSE @ F1-opt)",
                   color="#888", edgecolor="black", linewidth=0.6, zorder=2)
    bars2 = ax.bar(x + w/2, best_vals, w,
                   label="recommended (p99 + sensitivity-tuned)",
                   color="#1f77b4", edgecolor="black", linewidth=0.6, zorder=2)
    for bars, vals in ((bars1, base_vals), (bars2, best_vals)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(treatments)
    ax.set_xlabel("Treatment of the cardiologist's Unreadable label",
                  fontsize=10)
    ax.set_ylabel("Cohen's κ vs cardiologist (n = 48)", fontsize=10)
    ax.set_title(
        "Inter-rater agreement on the 48-beat blind set, before and after "
        "score-function and threshold improvements",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_xlim(-0.5, 2.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "8_cohens_kappa.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_score_comparison() -> None:
    """Side-by-side per-class sensitivity for mean-MSE vs p99 (priority classes)."""
    d = json.loads((RESULTS / "evaluate_p99.json").read_text())
    base_rows = {r["symbol"]: r for r in d["baseline_mean_mse"]["per_symbol"]}
    p99_rows  = {r["symbol"]: r for r in d["lever1_p99"]["per_symbol"]}

    priorities = [
        ("V", "PVC\n(wide-QRS)"),
        ("F", "Fusion\n(V + N)"),
        ("E", "Vent.\nescape"),
        ("f", "Fusion\n(paced + N)"),
        ("/", "Paced"),
        ("A", "APC"),
    ]
    base = [base_rows[s]["rate"] * 100 for s, _ in priorities]
    p99  = [p99_rows[s]["rate"]  * 100 for s, _ in priorities]
    labels = [lab for _, lab in priorities]

    x = np.arange(len(priorities))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, base, w, label="baseline (mean MSE)",
                   color="#888", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w/2, p99,  w, label="recommended (p99 of per-sample SE)",
                   color="#1f77b4", edgecolor="black", linewidth=0.6)
    for bars, vals in ((bars1, base), (bars2, p99)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 110)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sensitivity (%)", fontsize=10)
    ax.set_title(
        "Per-class detection sensitivity at the F1-optimal threshold — "
        "mean-MSE score vs p99 percentile-of-SE score",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "9_score_function_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    fig_per_class_rates()
    fig_kappa_levers()
    fig_score_comparison()


if __name__ == "__main__":
    main()
