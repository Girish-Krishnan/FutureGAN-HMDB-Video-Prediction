"""Visualize predictions from the trained model alongside original frames."""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import VideoDataset
from models import Generator
from torch.utils.data import DataLoader
from torchvision import transforms

def write_video(frames, filename, shape):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filename), fourcc, 30.0, shape)
    for frame in frames:
        out.write(cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR))
    out.release()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize predictions")
    parser.add_argument("--model", default="generator.pth", help="Generator checkpoint")
    parser.add_argument("--data_root", default="data/testing_data", help="Directory with test data")
    parser.add_argument("--video_subdir", default="video1", help="Sub-directory of frames to visualize")
    parser.add_argument("--output_original", default="original_video.mp4", help="Path for saving original video")
    parser.add_argument("--output_generated", default="generated_video.mp4", help="Path for saving generated video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.data_root) / args.video_subdir
    generator = Generator()
    generator.load_state_dict(torch.load(args.model))
    generator.eval()

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    testing_data = VideoDataset(args.data_root, transform)
    testing_loader = DataLoader(testing_data, batch_size=1, shuffle=False)

    original_frames = [
        cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        for frame in sorted(glob.glob(str(video_path / "*.jpg")))
    ]

    new_frames = []
    for i, (frame1, _) in enumerate(testing_loader):
        if i >= 1:
            break

        init_frame = frame1
        new_frames.append(init_frame[0].permute(1, 2, 0).clamp(0, 1))

        while len(new_frames) < len(original_frames):
            with torch.no_grad():
                prediction = generator(frame1)

            new_frames.append(prediction[0].permute(1, 2, 0).clamp(0, 1))
            frame1 = prediction

    write_video(original_frames, args.output_original, (320, 240))
    write_video(new_frames, args.output_generated, (64, 64))


if __name__ == "__main__":
    main()
