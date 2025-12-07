# models.py
"""Neural network architectures for ECG anomaly detection."""
from typing import Tuple

import torch
from torch import nn

from config import WINDOW_SAMPLES, LATENT_DIM


class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for ECG signals.

    Better than fully connected for time-series because:
    - Captures local temporal patterns (QRS complex, P/T waves)
    - Parameter efficient (weight sharing)
    - Translation invariant
    """

    def __init__(self, input_dim: int = WINDOW_SAMPLES, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input (batch, 1, 720) -> (batch, latent_dim)
        self.encoder = nn.Sequential(
            # (batch, 1, 720) -> (batch, 32, 360)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # (batch, 32, 360) -> (batch, 64, 180)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # (batch, 64, 180) -> (batch, 128, 90)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # (batch, 128, 90) -> (batch, 256, 45)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Flatten(),  # (batch, 256 * 45) = (batch, 11520)
        )

        # Calculate flattened size
        self._flat_size = 256 * (input_dim // 16)  # 11520 for 720 input

        # Bottleneck
        self.fc_encode = nn.Linear(self._flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._flat_size)

        # Decoder: (batch, latent_dim) -> (batch, 1, 720)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, input_dim // 16)),  # (batch, 256, 45)

            # (batch, 256, 45) -> (batch, 128, 90)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # (batch, 128, 90) -> (batch, 64, 180)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # (batch, 64, 180) -> (batch, 32, 360)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # (batch, 32, 360) -> (batch, 1, 720)
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)
        return self.fc_encode(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        h = self.fc_decode(z)
        out = self.decoder(h)
        return out.squeeze(1)  # (batch, 1, input_dim) -> (batch, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class DenseAutoencoder(nn.Module):
    """
    Fully connected autoencoder (original architecture, improved).

    Simpler but still effective for ECG anomaly detection.
    """

    def __init__(self, input_dim: int = WINDOW_SAMPLES, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for ECG signals.

    Advantages:
    - Better latent space structure
    - Can generate new samples
    - More robust anomaly detection via likelihood
    """

    def __init__(self, input_dim: int = WINDOW_SAMPLES, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss = Reconstruction loss + KL divergence.

    Args:
        recon_x: Reconstructed signal
        x: Original signal
        mu: Latent mean
        logvar: Latent log-variance
        beta: Weight for KL term (beta-VAE)
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
