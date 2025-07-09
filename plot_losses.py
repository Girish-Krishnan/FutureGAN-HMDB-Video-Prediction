"""Utility to plot generator and discriminator losses."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training losses")
    parser.add_argument("--loss_g", default="lossesG.npy", help="Path to generator losses")
    parser.add_argument("--loss_d", default="lossesD.npy", help="Path to discriminator losses")
    parser.add_argument("--output", default=None, help="Optional path to save the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lossesG = np.load(args.loss_g)
    lossesD = np.load(args.loss_d)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(lossesG, label="G")
    plt.plot(lossesD, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
