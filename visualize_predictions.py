import torch
import cv2
import numpy as np
from models import Generator
from dataset import VideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

def write_video(frames, filename, shape):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, shape)
    for frame in frames:
        out.write(cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == "__main__":
    # Define the paths
    video_path = './data/testing_data/video1/'
    original_video_path = './original_video.mp4'
    generated_video_path = './generated_video.mp4'

    # Load the generator
    generator = Generator()
    generator.load_state_dict(torch.load('generator.pth'))
    generator.eval()  # set to evaluation mode

    # Create a DataLoader for the testing data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # resize images to a manageable size
        transforms.ToTensor(),
    ])

    testing_data = VideoDataset("data/testing_data", transform)
    testing_loader = DataLoader(testing_data, batch_size=1, shuffle=False)

    # Load frames from the video_path directory
    original_frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) for frame in sorted(glob.glob(video_path + '*.jpg'))]

    # New frames will be the predicted frames
    new_frames = []

    # Generate the video
    for i, (frame1, frame2) in enumerate(testing_loader):
        if i >= 1: # only do the first video
            break
        
        init_frame = frame1
        new_frames.append(init_frame[0].permute(1, 2, 0).clamp(0, 1))  # clamp the values to be between 0 and 1

        while len(new_frames) < len(original_frames):
            with torch.no_grad():  # no need to calculate gradients
                prediction = generator(frame1)

            new_frames.append(prediction[0].permute(1, 2, 0).clamp(0, 1))  # clamp the values to be between 0 and 1
            frame1 = prediction

    # Write the frames to a video file
    write_video(original_frames, original_video_path, (320, 240))
    write_video(new_frames, generated_video_path, (64,64))
