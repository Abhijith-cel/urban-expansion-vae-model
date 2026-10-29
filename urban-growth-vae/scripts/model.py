"""
scripts/model.py
----------------
Sharpened Convolutional Variational Autoencoder (ConvVAE) and Dataset Loader
for Urban Satellite Imagery (128x128 RGB).

Key Enhancements:
- Hybrid L1 + MSE Reconstruction Loss for crisper, less-blurry building edges.
- 5-stage convolutional pyramid (down to 4x4 spatial resolution).
- Flatten dimension: 64 * 4 * 4 = 1,024 (memory-safe for CPU execution).
- Latent dimension: 32 with clamped log-variance to prevent numerical instability.
"""

import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset to load preprocessed 128x128 satellite images from data/processed/.
    Skips non-sample files (such as comparison.png).
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        all_pngs = sorted(glob(os.path.join(data_dir, "*.png")))
        self.image_paths = [p for p in all_pngs if not os.path.basename(p).startswith("comparison")]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL [0, 255] to Tensor [0.0, 1.0], shape (C, H, W)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor


def get_dataloader(data_dir: str, batch_size: int = 8, shuffle: bool = True):
    """Utility function to get a PyTorch DataLoader."""
    dataset = SatelliteDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ConvVAE(nn.Module):
    """
    Memory-efficient Convolutional VAE for 128x128 RGB images.
    - Input:  (Batch, 3, 128, 128)
    - Output: (Batch, 3, 128, 128) in range [0, 1]
    - Latent Dimension: 32
    """
    def __init__(self, latent_dim: int = 32):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # ----- ENCODER -----
        # Input: (B, 3, 128, 128)
        self.encoder = nn.Sequential(
            # Stage 1: 128x128 -> 64x64
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: 16x16 -> 8x8
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: 8x8 -> 4x4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64 channels * 4 * 4 = 1,024 features
        self.flatten_dim = 64 * 4 * 4

        # Latent heads: Mean and Log-Variance
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ----- DECODER -----
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # Stage 1: 4x4 -> 8x8
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: 8x8 -> 16x16
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: 64x64 -> 128x128
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Pixel output strictly in [0.0, 1.0]
        )

    def encode(self, x):
        """Encodes an image into latent mean and clamped log-variance."""
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes latent vector z into reconstructed 128x128 image."""
        h = self.decoder_input(z)
        h = h.view(-1, 64, 4, 4)
        recon_x = self.decoder(h)
        return recon_x

    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta: float = 0.0005):
    """
    Enhanced VAE loss:
    - 80% L1 Loss (Mean Absolute Error) for sharp edges and clear borders
    - 20% MSE Loss for smooth color fidelity
    - beta * KL Divergence for latent regularization
    """
    batch_size = x.size(0)
    # L1 Loss sharply penalizes blurred edges (unlike MSE which averages them)
    l1 = F.l1_loss(recon_x, x, reduction="sum") / batch_size
    mse = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    recon_loss = 0.8 * l1 + 0.2 * mse

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div
