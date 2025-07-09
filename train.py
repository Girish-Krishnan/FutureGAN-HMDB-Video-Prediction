"""Training script for FutureGAN."""

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
from evaluation_metrics import EvaluationMetrics
from models import Discriminator, Generator
import yaml

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FutureGAN")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--data_root",
        default="data/training_data",
        help="Directory with training frames",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (cpu, cuda, etc.). Defaults to auto-detection.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device_str = args.device
    if device_str is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    print("Using device:", device)
    metrics = EvaluationMetrics(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "generated_images"
    images_dir.mkdir(exist_ok=True)

    # Create data loaders
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    training_data = VideoDataset(args.data_root, transform)
    training_loader = DataLoader(
        training_data, batch_size=config["batch_size"], shuffle=True
    )

    # Initialize the networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"])
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"])

    # Loss logs
    lossesG: list[float] = []
    lossesD: list[float] = []

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Inception Score", "FID"])

    for epoch in range(config["epochs"]):
        progress_bar = tqdm(enumerate(training_loader), total=len(training_loader))
        for i, (frame1, real_frame2) in progress_bar:
            frame1 = frame1.to(device)
            real_frame2 = real_frame2.to(device)

            optimizerD.zero_grad()
            output = discriminator(frame1, real_frame2)
            errD_real = criterion(output, torch.ones_like(output))
            fake_frame2 = generator(frame1)
            output = discriminator(frame1.detach(), fake_frame2.detach())
            errD_fake = criterion(output, torch.zeros_like(output))
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            output = discriminator(frame1, fake_frame2)
            errG = criterion(output, torch.ones_like(output))
            errG.backward()
            optimizerG.step()

            progress_bar.set_description(
                f"Epoch {epoch + 1} [{i + 1}/{len(training_loader)}]..."
            )

        with torch.no_grad():
            real_images, _ = next(iter(training_loader))
            real_images = real_images.to(device)
            fake_images = generator(real_images).detach()

            grid_images = (
                make_grid(fake_images[:16], nrow=4, normalize=True)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            plt.imshow(grid_images)
            plt.axis("off")
            plt.savefig(images_dir / f"epoch_{epoch}.png")
            plt.close()

            inception_score_mean, inception_score_std = metrics.calculate_inception_score(
                fake_images
            )
            fid = metrics.calculate_frechet_inception_distance(real_images, fake_images)

        with metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, inception_score_mean, fid])

        lossesG.append(errG.item())
        lossesD.append(errD.item())
        print(f"Epoch: {epoch}, D loss: {errD.item()}, G loss: {errG.item()}")
        print(f"Inception score: {inception_score_mean} ± {inception_score_std}")
        print(f"FID: {fid}")

    np.save(output_dir / "lossesG.npy", np.array(lossesG))
    np.save(output_dir / "lossesD.npy", np.array(lossesD))
    torch.save(generator.state_dict(), output_dir / "generator.pth")


if __name__ == "__main__":
    main()
