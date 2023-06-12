import torch
from torchvision import transforms
from models import Generator
from dataset import VideoDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

# Generate predictions for the first 10 videos
for i, (frame1, frame2) in enumerate(testing_loader):
    if i >= 10:  # only do the first 10 videos
        break
    with torch.no_grad():  # no need to calculate gradients
        prediction = generator(frame1)
    
    # Display the input frame, the real next frame, and the predicted next frame
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(frame1[0].permute(1, 2, 0))
    plt.title("Input Frame")
    plt.subplot(1, 3, 2)
    plt.imshow(frame2[0].permute(1, 2, 0))
    plt.title("Real Next Frame")
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0].permute(1, 2, 0).clamp(0, 1))  # clamp the values to be between 0 and 1
    plt.title("Predicted Next Frame")
    plt.show()
