"""Training script for FutureGAN with new evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import VideoDataset
from evaluation_metrics import InceptionScore, FrechetInceptionDistance
from models import Discriminator, Generator
import yaml

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FutureGAN")
    parser.add_argument("--config", default="config.yaml", help="Path to a YAML configuration file")
    parser.add_argument("--data_root", default="data/training_data", help="Directory with training frames")
    parser.add_argument("--output_dir", default="outputs", help="Directory to write checkpoints and logs")
    parser.add_argument("--device", default=None, help="Device string (cpu, cuda, mps)")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device is not None:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print("Using device:", device)

    # Metrics
    inception_metric = InceptionScore(device)
    fid_metric = FrechetInceptionDistance(device)

    # Output dirs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "generated_images"
    img_dir.mkdir(exist_ok=True)

    # Data
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = VideoDataset(args.data_root, transform)
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

    # Models
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg["learning_rate"])
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg["learning_rate"])

    # Logs
    loss_g_hist: list[float] = []
    loss_d_hist: list[float] = []
    metrics_csv = out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "IS", "FID"])

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(cfg["epochs"]):
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, (frame1, real_frame2) in pbar:
            frame1 = frame1.to(device)
            real_frame2 = real_frame2.to(device)

            # --------------------------
            # Update Discriminator
            # --------------------------
            opt_d.zero_grad()
            out_real = disc(frame1, real_frame2)
            err_d_real = criterion(out_real, torch.ones_like(out_real))

            fake_frame2 = gen(frame1)
            out_fake = disc(frame1.detach(), fake_frame2.detach())
            err_d_fake = criterion(out_fake, torch.zeros_like(out_fake))

            err_d = err_d_real + err_d_fake
            err_d.backward()
            opt_d.step()

            # --------------------------
            # Update Generator
            # --------------------------
            opt_g.zero_grad()
            out_g = disc(frame1, fake_frame2)
            err_g = criterion(out_g, torch.ones_like(out_g))
            err_g.backward()
            opt_g.step()

            pbar.set_description(f"Epoch {epoch + 1} [{i + 1}/{len(loader)}]")

        # ------------------------------------------------------------------
        # Evaluation at epoch end
        # ------------------------------------------------------------------
        with torch.no_grad():
            real_batch, _ = next(iter(loader))
            real_batch = real_batch.to(device)
            fake_batch = gen(real_batch).detach()

            # Save grid for quick visual check
            grid = make_grid(fake_batch[:16], nrow=4, normalize=True).permute(1, 2, 0).cpu().numpy()
            plt.imshow(grid)
            plt.axis("off")
            plt.savefig(img_dir / f"epoch_{epoch}.png")
            plt.close()

            is_mean, is_std = inception_metric(fake_batch)
            fid_val = fid_metric(real_batch, fake_batch)

        # Log metrics
        with metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, is_mean, fid_val])

        loss_g_hist.append(err_g.item())
        loss_d_hist.append(err_d.item())
        print(f"Epoch {epoch + 1}: D {err_d.item():.4f}, G {err_g.item():.4f}, IS {is_mean:.3f}±{is_std:.3f}, FID {fid_val:.3f}")

    # ------------------------------------------------------------------
    # Save final artifacts
    # ------------------------------------------------------------------
    np.save(out_dir / "lossesG.npy", np.array(loss_g_hist))
    np.save(out_dir / "lossesD.npy", np.array(loss_d_hist))
    torch.save(gen.state_dict(), out_dir / "generator.pth")


if __name__ == "__main__":
    main()
