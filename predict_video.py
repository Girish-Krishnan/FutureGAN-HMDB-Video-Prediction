"""Generate a predicted video from a trained generator."""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from models import Generator

def load_model(model_path):
    model = Generator()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_image(img):
    # Convert the image from BGR to RGB, resize it, and normalize it.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # Assuming this is the input size of your model
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # Convert to torch tensor and rearrange dimensions
    return img

def write_video(frames, filename, size=(320, 240)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filename), fourcc, 30.0, size)
    for frame in frames:
        out.write(cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR))
    out.release()

def visualize_model(generator, initial_frames, num_predicted_frames):
    input_frame = initial_frames[-1]  # We don't want to modify the original sequence while predicting
    new_frames = []
    input_frame = process_image(input_frame).reshape(1,3,64,64).repeat(64, 1, 1, 1)

    for _ in range(num_predicted_frames):
        next_frame = generator(input_frame)  # Predict the next frame
        new_frames.append(next_frame)
        input_frame = next_frame  # Update the input frame to be the predicted frame

    return new_frames

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a predicted video")
    parser.add_argument("--model_path", default="generator.pth", help="Trained generator path")
    parser.add_argument("--video_path", default="./data/testing_data/video1/", help="Directory containing input frames")
    parser.add_argument("--output", default="predicted_video.mp4", help="Output video file")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = load_model(args.model_path)

    initial_frames = [
        cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        for frame in sorted(glob.glob(str(Path(args.video_path) / "*.jpg")))
    ]

    new_frames = visualize_model(generator, initial_frames, args.num_frames)
    new_frames = [
        np.array(frame.detach())[0, :, :, :].transpose(1, 2, 0) * 255
        for frame in new_frames
    ]
    new_frames = [cv2.resize(frame, (320, 240)) for frame in new_frames]
    new_frames = [
        cv2.copyMakeBorder(frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        for frame in new_frames
    ]

    frames = initial_frames + new_frames
    write_video(frames, args.output)


if __name__ == "__main__":
    main()
