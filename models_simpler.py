import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # output is an image
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),  # 6 = 3 (from frame1) + 3 (from frame2 or generated)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()  # output is a probability
        )

    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)
        return self.main(x)