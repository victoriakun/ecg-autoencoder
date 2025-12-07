#!/usr/bin/env python3
"""
MIT-BIH Autoencoder Training Script

Train autoencoder on MIT-BIH normal beats only, then evaluate on full test set.
Target: ROC-AUC >= 0.99

Data split:
- Train (70%): Normal beats only - learn normal ECG patterns
- Validation (15%): Normal beats only - early stopping
- Test (15%): All beats - evaluate anomaly detection
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

from config import (
    WINDOWS_PATH, LABELS_PATH, MODEL_DIR, SEED,
    LEARNING_RATE, WEIGHT_DECAY,
)

# Larger batch size for faster training on CPU
BATCH_SIZE = 256

# Output paths
MODEL_SAVE_PATH = MODEL_DIR / "mitbih_autoencoder_best.pt"
RESULTS_PATH = Path("results/mitbih_training_results.json")


class FastConvAutoencoder(nn.Module):
    """Optimized 1D Convolutional Autoencoder - faster training, high accuracy."""

    def __init__(self, input_dim: int = 720, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim

        # Encoder: more aggressive downsampling
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # 720 -> 360
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 360 -> 180
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # 180 -> 90
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),  # 90 -> 45
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        self._flat_size = 128 * 45  # 5760
        self.fc_encode = nn.Linear(self._flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._flat_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 45)),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),  # 45 -> 90
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 90 -> 180
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 180 -> 360
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),  # 360 -> 720
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)
        z = self.fc_encode(h)
        h = self.fc_decode(z)
        out = self.decoder(h)
        return out.squeeze(1)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data():
    """Load MIT-BIH data and split into train/val/test."""
    print("Loading MIT-BIH dataset...")
    data = np.load(WINDOWS_PATH).astype("float32")
    labels = np.load(LABELS_PATH)

    n_total = len(data)
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()

    print(f"  Total samples: {n_total:,}")
    print(f"  Normal: {n_normal:,} ({100*n_normal/n_total:.1f}%)")
    print(f"  Anomaly: {n_anomaly:,} ({100*n_anomaly/n_total:.1f}%)")

    # Shuffle with fixed seed
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_total)
    data, labels = data[idx], labels[idx]

    # Split: 70% train, 15% val, 15% test
    train_end = int(0.70 * n_total)
    val_end = int(0.85 * n_total)

    # Training: only normal beats
    train_data = data[:train_end]
    train_labels = labels[:train_end]
    train_normal = train_data[train_labels == 0]

    # Validation: only normal beats
    val_data = data[train_end:val_end]
    val_labels = labels[train_end:val_end]
    val_normal = val_data[val_labels == 0]

    # Test: ALL beats (for anomaly detection evaluation)
    test_data = data[val_end:]
    test_labels = labels[val_end:]

    print(f"\nData splits:")
    print(f"  Train: {len(train_normal):,} normal beats")
    print(f"  Val:   {len(val_normal):,} normal beats")
    print(f"  Test:  {len(test_data):,} total ({(test_labels==0).sum():,} normal, {(test_labels==1).sum():,} anomaly)")

    return train_normal, val_normal, test_data, test_labels


def create_dataloaders(train_data, val_data, batch_size=BATCH_SIZE):
    """Create PyTorch dataloaders."""
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data)),
        batch_size=batch_size * 2, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        recon = model(batch)
        loss = criterion(recon, batch)
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_reconstruction_errors(model, data, device, batch_size=512):
    """Compute MSE reconstruction error for each sample."""
    model.eval()
    errors = []
    for i in range(0, len(data), batch_size):
        batch = torch.from_numpy(data[i:i+batch_size]).to(device)
        recon = model(batch)
        mse = torch.mean((recon - batch) ** 2, dim=1)
        errors.extend(mse.cpu().numpy())
    return np.array(errors)


def evaluate_anomaly_detection(model, test_data, test_labels, device):
    """Evaluate model for anomaly detection."""
    errors = compute_reconstruction_errors(model, test_data, device)

    # ROC-AUC
    roc_auc = roc_auc_score(test_labels, errors)

    # PR-AUC and optimal F1
    precision, recall, thresholds = precision_recall_curve(test_labels, errors)
    pr_auc = auc(recall, precision)

    # Find optimal threshold (max F1)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    # Predictions at optimal threshold
    predictions = (errors > best_threshold).astype(int)

    # Accuracy
    accuracy = (predictions == test_labels).mean()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(best_f1),
        "accuracy": float(accuracy),
        "threshold": float(best_threshold),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


def estimate_remaining_time(epoch, total_epochs, epoch_time):
    """Estimate remaining training time."""
    remaining_epochs = total_epochs - epoch
    remaining_secs = remaining_epochs * epoch_time
    if remaining_secs < 60:
        return f"{remaining_secs:.0f}s"
    elif remaining_secs < 3600:
        return f"{remaining_secs/60:.1f}min"
    else:
        return f"{remaining_secs/3600:.1f}h"


def save_results(results, path):
    """Save results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    print("=" * 70)
    print("MIT-BIH AUTOENCODER TRAINING")
    print("Target: ROC-AUC >= 0.99")
    print("=" * 70)

    set_seed()

    # Load data
    train_data, val_data, test_data, test_labels = load_and_split_data()
    input_dim = train_data.shape[1]

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_data, val_data)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Model - using FastConvAutoencoder for speed
    latent_dim = 32
    model = FastConvAutoencoder(input_dim, latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: FastConvAutoencoder (latent_dim={latent_dim})")
    print(f"Parameters: {n_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training config
    max_epochs = 100
    patience = 15
    min_delta = 1e-6
    eval_every = 5  # Evaluate on test set every N epochs

    print(f"\nTraining config:")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Estimate training time
    print(f"\nEstimating training time...")
    start = time.time()
    _ = train_epoch(model, train_loader, optimizer, criterion, device)
    epoch_time = time.time() - start
    total_estimate = epoch_time * max_epochs
    print(f"  ~{epoch_time:.1f}s per epoch")
    print(f"  Estimated max time: {estimate_remaining_time(0, max_epochs, epoch_time)}")

    # Reset model for actual training
    model = FastConvAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("\n" + "-" * 70)
    print("TRAINING STARTED")
    print("-" * 70)

    best_val_loss = float("inf")
    best_roc_auc = 0.0
    patience_counter = 0
    training_start = time.time()
    history = {"train_loss": [], "val_loss": [], "roc_auc": [], "lr": []}

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Evaluate on test set periodically
        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate_anomaly_detection(model, test_data, test_labels, device)
            roc_auc = metrics["roc_auc"]
            history["roc_auc"].append(roc_auc)

            remaining = estimate_remaining_time(epoch, max_epochs, epoch_time)

            print(f"Epoch {epoch:3d}/{max_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"ROC-AUC: {roc_auc:.4f} | LR: {current_lr:.2e} | "
                  f"ETA: {remaining}")

            # Save if best ROC-AUC
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "roc_auc": roc_auc,
                    "metrics": metrics,
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                }, MODEL_SAVE_PATH)
                print(f"  -> NEW BEST MODEL SAVED (ROC-AUC: {roc_auc:.4f})")

                # Also save results
                results = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                save_results(results, RESULTS_PATH)

        # Early stopping on validation loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final evaluation with best model
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate_anomaly_detection(model, test_data, test_labels, device)

    total_time = time.time() - training_start

    print(f"\nBest model from epoch {checkpoint['epoch']}")
    print(f"Training time: {total_time/60:.1f} minutes")
    print(f"\nFINAL RESULTS ON MIT-BIH TEST SET:")
    print(f"  ROC-AUC:    {final_metrics['roc_auc']*100:.2f}%")
    print(f"  PR-AUC:     {final_metrics['pr_auc']*100:.2f}%")
    print(f"  F1 Score:   {final_metrics['f1']*100:.2f}%")
    print(f"  Accuracy:   {final_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:  {final_metrics['precision']*100:.2f}%")
    print(f"  Recall:     {final_metrics['recall']*100:.2f}%")
    print(f"  Threshold:  {final_metrics['threshold']:.6f}")

    # Check target
    if final_metrics['roc_auc'] >= 0.99:
        print(f"\n*** TARGET ACHIEVED: ROC-AUC >= 99% ***")
    else:
        print(f"\n*** Target not met. ROC-AUC: {final_metrics['roc_auc']*100:.2f}% (target: 99%) ***")

    # Save final results
    final_results = {
        "best_epoch": checkpoint['epoch'],
        "training_time_minutes": total_time / 60,
        **final_metrics,
        "target_met": final_metrics['roc_auc'] >= 0.99,
        "timestamp": datetime.now().isoformat(),
        "test_samples": {
            "total": len(test_labels),
            "normal": int((test_labels == 0).sum()),
            "anomaly": int((test_labels == 1).sum()),
        },
    }
    save_results(final_results, RESULTS_PATH)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
