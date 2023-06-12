import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Generator, Discriminator
from dataset import VideoDataset
import yaml
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load hyperparameters from config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
print("Using device: ", device)

# Create data loaders
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # resize images to a manageable size
    transforms.ToTensor(),
])
training_data = VideoDataset("data/training_data", transform)
training_loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)

# Initialize the networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(generator.parameters(), lr=config['learning_rate'])
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'])

# Loss logs
lossesG = []
lossesD = []

# Training loop
for epoch in range(config['epochs']):
    # Wrap the data loader with tqdm for a progress bar
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader))
    for i, (frame1, real_frame2) in progress_bar:
        # Move tensors to the device
        frame1 = frame1.to(device)
        real_frame2 = real_frame2.to(device)

        # Train discriminator
        optimizerD.zero_grad()
        output = discriminator(frame1, real_frame2)
        errD_real = criterion(output, torch.ones_like(output))
        fake_frame2 = generator(frame1)
        output = discriminator(frame1.detach(), fake_frame2.detach())
        errD_fake = criterion(output, torch.zeros_like(output))
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train generator
        optimizerG.zero_grad()
        output = discriminator(frame1, fake_frame2)
        errG = criterion(output, torch.ones_like(output))
        errG.backward()
        optimizerG.step()

        # Update progress bar
        progress_bar.set_description(f"Epoch {epoch+1} [{i+1}/{len(training_loader)}]...")

    # Log losses
    lossesG.append(errG.item())
    lossesD.append(errD.item())
    print(f"Epoch: {epoch}, D loss: {errD.item()}, G loss: {errG.item()}")

# Save losses
np.save('lossesG.npy', np.array(lossesG))
np.save('lossesD.npy', np.array(lossesD))

# Save the trained generator
torch.save(generator.state_dict(), 'generator.pth')
