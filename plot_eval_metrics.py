"""Utility for plotting evaluation metrics."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics")
    parser.add_argument(
        "--metrics",
        default="metrics.csv",
        help="Path to the metrics CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to store the generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, "r") as f:
        reader = csv.reader(f)
        next(reader)
        epochs, inception_scores, fids = zip(
            *[(int(r[0]), float(r[1]), float(r[2])) for r in reader]
        )

    plt.figure()
    plt.plot(epochs, inception_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Inception Score")
    plt.title("Inception Score over Epochs")
    plt.savefig(output_dir / "inception_score.png")

    plt.figure()
    plt.plot(epochs, fids)
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.title("FID over Epochs")
    plt.savefig(output_dir / "fid.png")
    plt.show()


if __name__ == "__main__":
    main()
