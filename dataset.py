from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
        self.frame_files = []
        for video_dir in self.video_dirs:
            frames = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            frames.sort()  # make sure frames are in order
            self.frame_files.extend(frames)

    def __len__(self):
        return len(self.frame_files) - 1  # we always get pairs of consecutive frames

    def __getitem__(self, idx):
        frame1 = Image.open(self.frame_files[idx]).convert("RGB")
        frame2 = Image.open(self.frame_files[idx+1]).convert("RGB")
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
        return frame1, frame2
