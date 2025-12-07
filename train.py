# train.py
"""Training script for ECG autoencoder models."""
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from config import (
    WINDOWS_PATH, MODEL_PATH, MODEL_DIR,
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TRAIN_RATIO, VAL_RATIO, PATIENCE, MIN_DELTA, LATENT_DIM,
)
from models import ConvAutoencoder, DenseAutoencoder, VariationalAutoencoder, vae_loss


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load and split data into train/val/test sets.

    IMPORTANT: Train and validation use ONLY normal beats.
    Test set contains both normal and anomaly for evaluation.
    """
    from config import LABELS_PATH

    data = np.load(WINDOWS_PATH).astype("float32")
    labels = np.load(LABELS_PATH)
    print(f"Loaded dataset: {data.shape}")
    print(f"  Normal: {(labels == 0).sum()}, Anomaly: {(labels == 1).sum()}")

    # Shuffle with consistent seed
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Split indices
    n = len(data)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    # For train and val: use ONLY normal beats
    train_data_all = data[:train_end]
    train_labels = labels[:train_end]
    train_data = train_data_all[train_labels == 0]  # Only normals!

    val_data_all = data[train_end:val_end]
    val_labels = labels[train_end:val_end]
    val_data = val_data_all[val_labels == 0]  # Only normals!

    # Test: keep ALL data (both normal and anomaly) for evaluation
    test_data = data[val_end:]

    print(f"Train: {len(train_data)} (normal only)")
    print(f"Val: {len(val_data)} (normal only)")
    print(f"Test: {len(test_data)} (mixed - for evaluation)")

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_data)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_data.shape[1]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_vae: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if is_vae:
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
        else:
            recon = model(batch)
            loss = criterion(recon, batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_vae: bool = False,
) -> float:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(device)

        if is_vae:
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
        else:
            recon = model(batch)
            loss = criterion(recon, batch)

        total_loss += loss.item() * len(batch)

    return total_loss / len(loader.dataset)


def train(
    model_type: str = "conv",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
) -> nn.Module:
    """
    Main training function.

    Args:
        model_type: "conv", "dense", or "vae"
        epochs: Number of training epochs
        lr: Learning rate
    """
    set_seed()

    train_loader, val_loader, test_loader, input_dim = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    is_vae = model_type == "vae"
    if model_type == "conv":
        model = ConvAutoencoder(input_dim, LATENT_DIM)
    elif model_type == "dense":
        model = DenseAutoencoder(input_dim, LATENT_DIM)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_dim, LATENT_DIM)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"Model: {model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_vae)
        val_loss = evaluate(model, val_loader, criterion, device, is_vae)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_type": model_type,
                "input_dim": input_dim,
                "latent_dim": LATENT_DIM,
            }, MODEL_PATH)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device, is_vae)
    print(f"\nFinal test loss: {test_loss:.6f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ECG autoencoder")
    parser.add_argument("--model", type=str, default="conv", choices=["conv", "dense", "vae"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    train(model_type=args.model, epochs=args.epochs, lr=args.lr)
