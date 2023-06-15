import torch
import cv2
import numpy as np
from models import Generator
import glob

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

def write_video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (320, 240))
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

if __name__ == "__main__":
    # Define the paths
    model_path = 'generator.pth'
    video_path = './data/testing_data/video1/'
    output_path = 'predicted_video.mp4'

    # Load the model
    generator = load_model(model_path)

    # Load frames from the video_path directory
    initial_frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) for frame in sorted(glob.glob(video_path + '*.jpg'))]

    # Generate the video
    num_predicted_frames = 20  # Number of frames to generate
    new_frames = visualize_model(generator, initial_frames, num_predicted_frames)
    new_frames = [np.array(frame.detach())[0,:,:,:].transpose(1, 2, 0) * 255 for frame in new_frames] # Convert frames back to numpy arrays with shape (H, W, C) and range [0, 255]
    # Interpolate new frames to 320x240
    new_frames = [cv2.resize(frame, (320, 240)) for frame in new_frames]

    # Add a red border to each image in new_frames, without changing the dimensions
    new_frames = [cv2.copyMakeBorder(frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 0, 0]) for frame in new_frames]

    # Write the frames to a video file
    frames =  initial_frames + new_frames
    write_video(frames, output_path)
