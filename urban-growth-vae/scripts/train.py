"""
scripts/train.py
----------------
Enhanced Training Pipeline for Urban Growth Variational Autoencoder (ConvVAE).

Features:
- Hybrid L1 + MSE loss for sharp, high-contrast reconstructions
- Learning rate scheduler for fine detail convergence
- Checkpoint persistence in models/vae_best.pth
- Reconstructed sample visualization in outputs/reconstructions/
- Training loss curve plot in outputs/loss_curve.png
"""

import os
import gc
import random
import argparse
import tempfile
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Safe Matplotlib configuration for cache
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.pyplot as plt

from model import ConvVAE, SatelliteDataset, vae_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RECON_DIR = os.path.join(OUTPUTS_DIR, "reconstructions")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_reconstruction_plot(original, reconstructed, save_path: str, n_samples: int = 4):
    n = min(n_samples, original.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        orig_img = original[i].detach().cpu().permute(1, 2, 0).numpy()
        recon_img = reconstructed[i].detach().cpu().permute(1, 2, 0).numpy()

        axes[i, 0].imshow(np.clip(orig_img, 0, 1))
        axes[i, 0].set_title(f"Original #{i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(recon_img, 0, 1))
        axes[i, 1].set_title(f"Reconstructed #{i+1}")
        axes[i, 1].axis("off")

    plt.suptitle("Satellite Image VAE Reconstruction (Sharpened Loss)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Reconstructed samples saved to: {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path: str):
    plt.figure(figsize=(8, 4.5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss", markersize=3)
    plt.plot(epochs, val_losses, "r--s", label="Validation Loss", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("VAE Total Loss (L1 + MSE + KL)")
    plt.title("Urban Growth VAE Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve plot saved to: {save_path}")


def train(epochs: int = 40, batch_size: int = 8, lr: float = 1e-3, latent_dim: int = 32, beta: float = 0.0005, seed: int = 42):
    set_seed(seed)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")

    # 1. Dataset loading & splitting
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(f"Processed dataset directory not found at: {PROCESSED_DIR}")

    full_dataset = SatelliteDataset(PROCESSED_DIR)
    total_images = len(full_dataset)
    if total_images < 2:
        raise ValueError(f"Insufficient images found in {PROCESSED_DIR}. Found: {total_images}")

    val_size = max(1, int(total_images * 0.2))
    train_size = total_images - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Dataset loaded: Total={total_images} | Train={train_size} | Validation={val_size}")

    # 2. Model & Optimizer initialization
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(MODELS_DIR, "vae_best.pth")

    print(f"[INFO] Starting training for {epochs} epochs (Batch size={batch_size}, LR={lr}, Beta={beta}, Latent={latent_dim})...\n")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon_x, mu, logvar = model(batch)
            loss, recon_l, kl_l = vae_loss(recon_x, batch, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)
            train_recon += recon_l.item() * batch.size(0)
            train_kl += kl_l.item() * batch.size(0)

            del batch, recon_x, mu, logvar, loss

        scheduler.step()

        epoch_train_loss = train_loss / train_size
        epoch_train_recon = train_recon / train_size
        epoch_train_kl = train_kl / train_size
        train_loss_history.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        sample_val_batch = None
        sample_val_recon = None

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_x, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon_x, batch, mu, logvar, beta=beta)

                val_loss += loss.item() * batch.size(0)
                val_recon += recon_l.item() * batch.size(0)
                val_kl += kl_l.item() * batch.size(0)

                if sample_val_batch is None:
                    sample_val_batch = batch.clone()
                    sample_val_recon = recon_x.clone()

                del batch, recon_x, mu, logvar, loss

        epoch_val_loss = val_loss / val_size
        epoch_val_recon = val_recon / val_size
        epoch_val_kl = val_kl / val_size
        val_loss_history.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:7.2f} (Recon: {epoch_train_recon:7.2f}, KL: {epoch_train_kl:5.2f}) | "
                f"Val Loss: {epoch_val_loss:7.2f} (Recon: {epoch_val_recon:7.2f}, KL: {epoch_val_kl:5.2f})"
            )

        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "latent_dim": latent_dim
            }, best_checkpoint_path)

        gc.collect()

    print(f"\n[SUCCESS] Training finished. Best Validation Loss: {best_val_loss:.2f}")

    # 4. Save Final Reconstruction Samples
    recon_sample_path = os.path.join(RECON_DIR, f"reconstructions_epoch_{epochs}.png")
    if sample_val_batch is not None and sample_val_recon is not None:
        save_reconstruction_plot(sample_val_batch, sample_val_recon, recon_sample_path, n_samples=4)

    # 5. Save Loss Curve
    loss_curve_path = os.path.join(OUTPUTS_DIR, "loss_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, loss_curve_path)

    # 6. Verify Checkpoint Loading
    print("\n[INFO] Verifying saved checkpoint integrity...")
    test_model = ConvVAE(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    test_model.load_state_dict(checkpoint["model_state_dict"])
    test_model.eval()

    with torch.no_grad():
        test_in = torch.randn(1, 3, 128, 128).to(device)
        test_out, _, _ = test_model(test_in)
        assert test_out.shape == (1, 3, 128, 128), "Reloaded model output shape mismatch!"

    print(f"[VERIFIED] Checkpoint successfully reloaded from: {best_checkpoint_path}")
    print("[PHASE 5 COMPLETE] Training completed with sharpened loss functions!")


def main():
    parser = argparse.ArgumentParser(description="Train VAE on Satellite Images.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent vector dimension (default: 32)")
    parser.add_argument("--beta", type=float, default=0.0005, help="KL loss weight (default: 0.0005)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        beta=args.beta,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
