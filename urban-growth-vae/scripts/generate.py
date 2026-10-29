"""
scripts/generate.py
-------------------
AI-Powered Urban Growth Simulation & Scenario Generation Pipeline (ConvVAE).

Modes:
1. Growth Simulation (--mode growth, default):
   Displays [Original Satellite Image] alongside 4 progression stages of urban expansion.
2. Candidate Scenarios (--mode scenarios):
   Displays [Original Satellite Image] alongside baseline reconstruction and candidate scenarios.
"""

import os
import argparse
import tempfile
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib.pyplot as plt

from model import ConvVAE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GENERATED_DIR = os.path.join(OUTPUTS_DIR, "generated")
GROWTH_DIR = os.path.join(OUTPUTS_DIR, "growth_simulation")


def run_scenarios_mode(image_path: str, checkpoint_path: str, num_scenarios: int = 3, show_display: bool = True):
    """Reverts to the previous mode: Original Input alongside generated candidate scenarios."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConvVAE(latent_dim=checkpoint.get("latent_dim", 32)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Image
    raw_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    scenarios = []
    labels = []

    with torch.no_grad():
        mu, logvar = model.encode(input_tensor)

        # Baseline Reconstruction
        base_recon = model.decode(mu)
        scenarios.append(base_recon.squeeze(0).cpu().permute(1, 2, 0).numpy())
        labels.append("Baseline VAE\nReconstruction")

        # Candidate Scenarios
        torch.manual_seed(42)
        shifts = [0.5, 1.0, 1.5, 2.0]
        for i in range(num_scenarios):
            strength = shifts[i % len(shifts)]
            z_s = mu + strength * torch.randn_like(mu)
            gen = model.decode(z_s).squeeze(0).cpu().permute(1, 2, 0).numpy()
            scenarios.append(np.clip(gen, 0.0, 1.0))
            labels.append(f"Scenario #{i+1}\n(Shift $\\sigma$={strength})")

            # Save individual image
            out_f = os.path.join(GENERATED_DIR, f"scenario_{i+1}.png")
            Image.fromarray((np.clip(gen, 0.0, 1.0) * 255).astype(np.uint8)).save(out_f)

    # Plot
    total = 1 + len(scenarios)
    fig, axes = plt.subplots(1, total, figsize=(3.3 * total, 3.8))

    orig_np = np.array(raw_img) / 255.0
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original Input\n({os.path.basename(image_path)})", fontweight="bold")
    axes[0].axis("off")

    for idx, (img, lbl) in enumerate(zip(scenarios, labels), start=1):
        axes[idx].imshow(img)
        axes[idx].set_title(lbl, fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("Urban Satellite Layout Scenario Generation (ConvVAE)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_plot = os.path.join(GENERATED_DIR, "generated_comparison.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"[INFO] Comparison saved to: {out_plot}")

    if show_display:
        print("[INFO] Displaying comparison window (close window to finish)...")
        plt.show()
    else:
        plt.close()


def run_growth_mode(start_path: str, target_path: str, checkpoint_path: str, num_stages: int = 4, show_display: bool = True):
    """Growth simulation mode with Original Input prominently displayed on the left."""
    os.makedirs(GROWTH_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConvVAE(latent_dim=checkpoint.get("latent_dim", 32)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_start = Image.open(start_path).convert("RGB")
    raw_target = Image.open(target_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])

    img_start = transform(raw_start).unsqueeze(0).to(device)
    img_target = transform(raw_target).unsqueeze(0).to(device)

    with torch.no_grad():
        mu_start, _ = model.encode(img_start)
        mu_target, _ = model.encode(img_target)

        alphas = np.linspace(0.0, 1.0, num_stages)
        stage_imgs = []
        stage_lbls = []

        for idx, alpha in enumerate(alphas):
            z_t = (1.0 - alpha) * mu_start + alpha * mu_target
            gen = model.decode(z_t).squeeze(0).cpu().permute(1, 2, 0).numpy()
            gen = np.clip(gen, 0.0, 1.0)
            stage_imgs.append(gen)
            pct = int(alpha * 100)
            stage_lbls.append(f"Expansion Stage {idx+1}\n({pct}% Density)")

            out_f = os.path.join(GROWTH_DIR, f"stage_{idx+1}.png")
            Image.fromarray((gen * 255).astype(np.uint8)).save(out_f)

    # Plot: [Original Input] + [Stages 1..N]
    total = 1 + len(stage_imgs)
    fig, axes = plt.subplots(1, total, figsize=(3.2 * total, 4.0))

    orig_np = np.array(raw_start) / 255.0
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original Input\n({os.path.basename(start_path)})", fontweight="bold")
    axes[0].axis("off")

    for idx, (img, lbl) in enumerate(zip(stage_imgs, stage_lbls), start=1):
        axes[idx].imshow(img)
        axes[idx].set_title(lbl, fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("AI Urban Expansion Prediction & City Growth Simulation (ConvVAE)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_plot = os.path.join(GROWTH_DIR, "urban_growth_timelapse.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"[INFO] Simulation comparison saved to: {out_plot}")

    if show_display:
        print("[INFO] Displaying simulation window (close window to finish)...")
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Urban Growth VAE Generator.")
    parser.add_argument("--mode", type=str, default=None, choices=["scenarios", "growth"], help="Display mode: 'scenarios' or 'growth'")
    parser.add_argument("--image", "--start", dest="image", type=str, default=None, help="Input / Start satellite image path")
    parser.add_argument("--target", type=str, default=None, help="Target high-density image path for growth mode")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(MODELS_DIR, "vae_best.pth"))
    parser.add_argument("--num-scenarios", type=int, default=3, help="Number of scenarios")
    parser.add_argument("--stages", type=int, default=4, help="Number of growth stages")
    parser.add_argument("--no-display", action="store_true", help="Run without popup window")
    args = parser.parse_args()

    # Automatically infer mode: if --target or --start/--image with --stages is specified, use growth mode
    if args.mode is None:
        if args.target is not None:
            mode = "growth"
        else:
            mode = "scenarios"
    else:
        mode = args.mode

    # Default images if not specified
    all_imgs = [f for f in sorted(os.listdir(PROCESSED_DIR)) if f.endswith(".png") and not f.startswith("comparison")]
    start_img = args.image if args.image else os.path.join(PROCESSED_DIR, all_imgs[0])
    target_img = args.target if args.target else os.path.join(PROCESSED_DIR, all_imgs[-1])

    if mode == "scenarios":
        run_scenarios_mode(start_img, args.checkpoint, num_scenarios=args.num_scenarios, show_display=not args.no_display)
    else:
        run_growth_mode(start_img, target_img, args.checkpoint, num_stages=args.stages, show_display=not args.no_display)


if __name__ == "__main__":
    main()
