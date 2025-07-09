"""Simple script for generating predictions on test data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VideoDataset
from models import Generator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained generator")
    parser.add_argument("--model", default="generator.pth", help="Path to model checkpoint")
    parser.add_argument("--data_root", default="data/testing_data", help="Path to testing data")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to visualize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = Generator()
    generator.load_state_dict(torch.load(args.model))
    generator.eval()

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    testing_data = VideoDataset(args.data_root, transform)
    testing_loader = DataLoader(testing_data, batch_size=1, shuffle=False)

    for i, (frame1, frame2) in enumerate(testing_loader):
        if i >= args.num_examples:
            break
        with torch.no_grad():
            prediction = generator(frame1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(frame1[0].permute(1, 2, 0))
        plt.title("Input Frame")
        plt.subplot(1, 3, 2)
        plt.imshow(frame2[0].permute(1, 2, 0))
        plt.title("Real Next Frame")
        plt.subplot(1, 3, 3)
        plt.imshow(prediction[0].permute(1, 2, 0).clamp(0, 1))
        plt.title("Predicted Next Frame")
        plt.show()


if __name__ == "__main__":
    main()
