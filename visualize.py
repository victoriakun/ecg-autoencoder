# visualize.py
"""Visualization tools for ECG anomaly detection."""
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    MODEL_PATH, WINDOWS_PATH, LABELS_PATH, SEED,
    TRAIN_RATIO, VAL_RATIO, LATENT_DIM, SAMPLING_RATE,
)
from models import ConvAutoencoder, DenseAutoencoder, VariationalAutoencoder


def load_model_and_data(device: torch.device):
    """Load model and test data."""
    # Load data
    X = np.load(WINDOWS_PATH).astype("float32")
    y = np.load(LABELS_PATH)

    # Same split as training
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_test, y_test = X[test_start:], y[test_start:]

    # Load model
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

    return model, model_type, X_test, y_test


def plot_single_reconstruction(
    model: torch.nn.Module,
    x: np.ndarray,
    label: int,
    model_type: str,
    device: torch.device,
    ax: plt.Axes = None,
) -> float:
    """Plot original vs reconstructed signal."""
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "vae":
            x_recon, _, _ = model(x_tensor)
        else:
            x_recon = model(x_tensor)
        x_recon = x_recon.squeeze().cpu().numpy()

    error = np.mean((x - x_recon) ** 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    time = np.arange(len(x)) / SAMPLING_RATE  # Convert to seconds

    ax.plot(time, x, label="Original", alpha=0.8, linewidth=1.5)
    ax.plot(time, x_recon, label="Reconstructed", alpha=0.8, linewidth=1.5)
    ax.fill_between(time, x, x_recon, alpha=0.3, color="red", label="Error")

    label_str = "ANOMALY" if label == 1 else "NORMAL"
    ax.set_title(f"{label_str} | MSE: {error:.6f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return error


def plot_multiple_reconstructions(n_samples: int = 8, save_path: Optional[str] = None) -> None:
    """Plot multiple reconstructions: normals and anomalies side by side."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, X_test, y_test = load_model_and_data(device)

    # Get indices for normals and anomalies
    normal_idx = np.where(y_test == 0)[0]
    anomaly_idx = np.where(y_test == 1)[0]

    n_each = n_samples // 2
    rng = np.random.default_rng(SEED + 1)

    selected_normal = rng.choice(normal_idx, size=min(n_each, len(normal_idx)), replace=False)
    selected_anomaly = rng.choice(anomaly_idx, size=min(n_each, len(anomaly_idx)), replace=False)

    fig, axes = plt.subplots(n_each, 2, figsize=(16, 3 * n_each))
    fig.suptitle("Reconstruction Comparison: Normal vs Anomaly", fontsize=14, fontweight="bold")

    for i in range(n_each):
        # Normal
        if i < len(selected_normal):
            idx = selected_normal[i]
            plot_single_reconstruction(model, X_test[idx], y_test[idx], model_type, device, axes[i, 0])
            axes[i, 0].set_ylabel(f"Sample {i+1}")

        # Anomaly
        if i < len(selected_anomaly):
            idx = selected_anomaly[i]
            plot_single_reconstruction(model, X_test[idx], y_test[idx], model_type, device, axes[i, 1])

    axes[0, 0].set_title("NORMAL Beats", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("ANOMALY Beats", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_error_distribution(save_path: Optional[str] = None) -> None:
    """Plot reconstruction error distribution for normal vs anomaly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, X_test, y_test = load_model_and_data(device)

    # Compute all reconstruction errors
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            batch = torch.from_numpy(X_test[i:i+256]).to(device)
            if model_type == "vae":
                recon, _, _ = model(batch)
            else:
                recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    errors = np.concatenate(errors)

    normal_errors = errors[y_test == 0]
    anomaly_errors = errors[y_test == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(normal_errors, bins=50, alpha=0.7, label=f"Normal (n={len(normal_errors)})",
            color="blue", density=True)
    ax.hist(anomaly_errors, bins=50, alpha=0.7, label=f"Anomaly (n={len(anomaly_errors)})",
            color="red", density=True)

    # Threshold lines
    threshold_3std = normal_errors.mean() + 3 * normal_errors.std()
    threshold_p99 = np.percentile(normal_errors, 99)

    ax.axvline(threshold_3std, color="green", linestyle="--", lw=2, label=f"mean+3std ({threshold_3std:.4f})")
    ax.axvline(threshold_p99, color="orange", linestyle="--", lw=2, label=f"p99 ({threshold_p99:.4f})")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([normal_errors, anomaly_errors], labels=["Normal", "Anomaly"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")

    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_title("Error Distribution (Box Plot)")
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (
        f"Normal: mean={normal_errors.mean():.4f}, std={normal_errors.std():.4f}\n"
        f"Anomaly: mean={anomaly_errors.mean():.4f}, std={anomaly_errors.std():.4f}\n"
        f"Separation: {(anomaly_errors.mean() - normal_errors.mean()) / normal_errors.std():.2f} std"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_worst_and_best(n_samples: int = 5, save_path: Optional[str] = None) -> None:
    """
    Plot samples with highest and lowest reconstruction errors.
    Useful for error analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, X_test, y_test = load_model_and_data(device)

    # Compute errors
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            batch = torch.from_numpy(X_test[i:i+256]).to(device)
            if model_type == "vae":
                recon, _, _ = model(batch)
            else:
                recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    errors = np.concatenate(errors)

    # Find worst (highest error) and best (lowest error)
    worst_idx = np.argsort(errors)[-n_samples:][::-1]
    best_idx = np.argsort(errors)[:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 3 * n_samples))
    fig.suptitle("Best vs Worst Reconstructions", fontsize=14, fontweight="bold")

    for i in range(n_samples):
        # Best (lowest error)
        idx = best_idx[i]
        plot_single_reconstruction(model, X_test[idx], y_test[idx], model_type, device, axes[i, 0])
        axes[i, 0].set_title(f"BEST #{i+1} | {'Anomaly' if y_test[idx] else 'Normal'} | MSE: {errors[idx]:.6f}")

        # Worst (highest error)
        idx = worst_idx[i]
        plot_single_reconstruction(model, X_test[idx], y_test[idx], model_type, device, axes[i, 1])
        axes[i, 1].set_title(f"WORST #{i+1} | {'Anomaly' if y_test[idx] else 'Normal'} | MSE: {errors[idx]:.6f}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_latent_space(save_path: Optional[str] = None) -> None:
    """
    Visualize the latent space using t-SNE.
    Shows how normal and anomaly samples cluster.
    """
    from sklearn.manifold import TSNE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, X_test, y_test = load_model_and_data(device)

    # Get latent representations
    latent_vectors = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            batch = torch.from_numpy(X_test[i:i+256]).to(device)
            if model_type == "vae":
                mu, _ = model.encode(batch)
                z = mu  # Use mean for VAE
            else:
                z = model.encode(batch)
            latent_vectors.append(z.cpu().numpy())

    latent = np.vstack(latent_vectors)
    print(f"Latent space shape: {latent.shape}")

    # Subsample for t-SNE (it's slow on large datasets)
    max_samples = 5000
    if len(latent) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(latent), size=max_samples, replace=False)
        latent_sub = latent[idx]
        y_sub = y_test[idx]
    else:
        latent_sub = latent
        y_sub = y_test

    print(f"Running t-SNE on {len(latent_sub)} samples...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    latent_2d = tsne.fit_transform(latent_sub)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    normal_mask = y_sub == 0
    anomaly_mask = y_sub == 1

    ax.scatter(latent_2d[normal_mask, 0], latent_2d[normal_mask, 1],
               c="blue", alpha=0.5, s=10, label=f"Normal (n={normal_mask.sum()})")
    ax.scatter(latent_2d[anomaly_mask, 0], latent_2d[anomaly_mask, 1],
               c="red", alpha=0.7, s=20, label=f"Anomaly (n={anomaly_mask.sum()})")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Latent Space Visualization ({model_type} model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_training_history(log_path: str = "training_log.csv", save_path: Optional[str] = None) -> None:
    """Plot training history if log file exists."""
    import pandas as pd

    if not Path(log_path).exists():
        print(f"Training log not found: {log_path}")
        print("To generate a log, modify train.py to save losses to CSV.")
        return

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    if "lr" in df.columns:
        ax = axes[1]
        ax.plot(df["epoch"], df["lr"], linewidth=2, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_model_comparison(models_dir: str = "models", save_path: Optional[str] = None) -> None:
    """
    Compare multiple trained models.
    Expects models saved as: models/ecg_conv.pt, models/ecg_dense.pt, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    X = np.load(WINDOWS_PATH).astype("float32")
    y = np.load(LABELS_PATH)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]
    test_start = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_test, y_test = X[test_start:], y[test_start:]

    results = {}
    model_files = list(Path(models_dir).glob("*.pt"))

    if not model_files:
        print(f"No model files found in {models_dir}")
        return

    for model_path in model_files:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model_type = checkpoint.get("model_type", model_path.stem)
            input_dim = checkpoint["input_dim"]
            latent_dim = checkpoint.get("latent_dim", LATENT_DIM)

            if "conv" in model_type:
                model = ConvAutoencoder(input_dim, latent_dim)
            elif "dense" in model_type:
                model = DenseAutoencoder(input_dim, latent_dim)
            elif "vae" in model_type:
                model = VariationalAutoencoder(input_dim, latent_dim)
            else:
                continue

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Compute errors
            errors = []
            is_vae = "vae" in model_type
            with torch.no_grad():
                for i in range(0, len(X_test), 256):
                    batch = torch.from_numpy(X_test[i:i+256]).to(device)
                    if is_vae:
                        recon, _, _ = model(batch)
                    else:
                        recon = model(batch)
                    mse = torch.mean((recon - batch) ** 2, dim=1)
                    errors.append(mse.cpu().numpy())

            errors = np.concatenate(errors)
            results[model_type] = errors
            print(f"Loaded {model_type}: mean_error={errors.mean():.6f}")

        except Exception as e:
            print(f"Failed to load {model_path}: {e}")

    if not results:
        print("No models loaded successfully")
        return

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, errors in results.items():
        normal_errors = errors[y_test == 0]
        anomaly_errors = errors[y_test == 1]

        # Calculate separation
        separation = (anomaly_errors.mean() - normal_errors.mean()) / normal_errors.std()

        ax.hist(errors[y_test == 0], bins=50, alpha=0.5, density=True,
                label=f"{model_name} Normal")
        ax.hist(errors[y_test == 1], bins=50, alpha=0.5, density=True,
                label=f"{model_name} Anomaly (sep={separation:.2f}σ)")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Model Comparison: Error Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize ECG anomaly detection results")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["all", "recon", "error", "worst", "latent"],
                        help="Which plot to generate")
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    args = parser.parse_args()

    save_dir = Path("figures")
    if args.save:
        save_dir.mkdir(exist_ok=True)

    if args.plot in ["all", "recon"]:
        print("\n=== Reconstruction Examples ===")
        plot_multiple_reconstructions(
            n_samples=8,
            save_path=str(save_dir / "reconstructions.png") if args.save else None
        )

    if args.plot in ["all", "error"]:
        print("\n=== Error Distribution ===")
        plot_error_distribution(
            save_path=str(save_dir / "error_distribution.png") if args.save else None
        )

    if args.plot in ["all", "worst"]:
        print("\n=== Best & Worst Reconstructions ===")
        plot_worst_and_best(
            n_samples=5,
            save_path=str(save_dir / "best_worst.png") if args.save else None
        )

    if args.plot in ["all", "latent"]:
        print("\n=== Latent Space ===")
        plot_latent_space(
            save_path=str(save_dir / "latent_space.png") if args.save else None
        )
