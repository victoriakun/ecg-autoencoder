# evaluate.py
"""Evaluation and anomaly detection for trained models."""
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from config import (
    MODEL_PATH, WINDOWS_PATH, LABELS_PATH, SEED,
    TRAIN_RATIO, VAL_RATIO, LATENT_DIM,
)
from models import ConvAutoencoder, DenseAutoencoder, VariationalAutoencoder


def load_model(device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", "conv")
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint.get("latent_dim", LATENT_DIM)

    if model_type == "conv":
        model = ConvAutoencoder(input_dim, latent_dim)
    elif model_type == "dense":
        model = DenseAutoencoder(input_dim, latent_dim)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    return model, model_type


def compute_reconstruction_errors(
    model: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    is_vae: bool = False,
) -> np.ndarray:
    """Compute reconstruction MSE for all samples."""
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[i : i + batch_size]).to(device)

            if is_vae:
                recon, _, _ = model(batch)
            else:
                recon = model(batch)

            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors)


def find_optimal_threshold(errors: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal threshold using various methods.

    Returns:
        threshold: Optimal threshold value
        metrics: Dictionary of metrics at optimal threshold
    """
    # Method 1: Maximize F1 score
    precision, recall, thresholds_pr = precision_recall_curve(labels, errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else thresholds_pr[-1]

    # Method 2: Statistical (mean + k*std on normal samples only)
    normal_errors = errors[labels == 0]
    threshold_3std = normal_errors.mean() + 3 * normal_errors.std()

    # Method 3: Percentile on normal samples
    threshold_p99 = np.percentile(normal_errors, 99)

    # Use F1-optimal as default
    predictions = (errors > best_threshold_f1).astype(int)

    metrics = {
        "threshold_f1": best_threshold_f1,
        "threshold_3std": threshold_3std,
        "threshold_p99": threshold_p99,
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

    return best_threshold_f1, metrics


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    is_vae: bool = False,
) -> dict:
    """
    Complete evaluation of the anomaly detection model.

    Args:
        model: Trained autoencoder
        X_test: Test windows
        y_test: Test labels (0=normal, 1=anomaly)
        device: torch device
        is_vae: Whether model is VAE

    Returns:
        Dictionary with all metrics and optimal threshold
    """
    errors = compute_reconstruction_errors(model, X_test, device, is_vae=is_vae)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, errors)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, errors)
    pr_auc = auc(recall, precision)

    # Find optimal threshold
    threshold, threshold_metrics = find_optimal_threshold(errors, y_test)

    # Final predictions
    predictions = (errors > threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, predictions)

    results = {
        "errors": errors,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision_curve": precision,
        "recall_curve": recall,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": conf_matrix,
        **threshold_metrics,
    }

    return results


def plot_results(results: Dict, y_test: np.ndarray, save_path: Optional[str] = None) -> None:
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ROC Curve
    ax = axes[0, 0]
    ax.plot(results["fpr"], results["tpr"], "b-", lw=2, label=f'ROC (AUC = {results["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    ax.plot(results["recall_curve"], results["precision_curve"], "g-", lw=2,
            label=f'PR (AUC = {results["pr_auc"]:.3f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # 3. Error Distribution
    ax = axes[1, 0]
    errors = results["errors"]
    normal_errors = errors[y_test == 0]
    anomaly_errors = errors[y_test == 1]

    ax.hist(normal_errors, bins=50, alpha=0.6, label="Normal", color="blue", density=True)
    ax.hist(anomaly_errors, bins=50, alpha=0.6, label="Anomaly", color="red", density=True)
    ax.axvline(results["threshold"], color="green", linestyle="--", lw=2, label=f'Threshold ({results["threshold"]:.4f})')
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    ax = axes[1, 1]
    cm = results["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def print_report(results: Dict, y_test: np.ndarray) -> None:
    """Print detailed evaluation report."""
    predictions = (results["errors"] > results["threshold"]).astype(int)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOptimal Threshold: {results['threshold']:.6f}")
    print(f"  (Alternative: 3*std = {results['threshold_3std']:.6f}, p99 = {results['threshold_p99']:.6f})")

    print(f"\nMetrics at optimal threshold:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")

    print(f"\nCurve-based metrics:")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"  PR-AUC:    {results['pr_auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Normal", "Anomaly"]))


def main() -> None:
    """Main evaluation function."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    X = np.load(WINDOWS_PATH).astype("float32")
    y = np.load(LABELS_PATH)

    # Use same split as training
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_test, y_test = X[test_start:], y[test_start:]

    print(f"Test set: {len(X_test)} samples ({np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomaly)")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(device)
    is_vae = model_type == "vae"

    # Evaluate
    results = evaluate_model(model, X_test, y_test, device, is_vae)

    # Print report
    print_report(results, y_test)

    # Plot results
    plot_results(results, y_test, save_path="evaluation_results.png")


if __name__ == "__main__":
    main()
