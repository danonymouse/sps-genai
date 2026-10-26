"""GAN for MNIST digits (Assignment 3, Module 6): Generator and Discriminator."""

import torch
import torch.nn as nn

LATENT_DIM = 100  # size of the random noise vector the Generator starts from


class Generator(nn.Module):
    """Turns a noise vector (batch, 100) into a 28x28 grayscale image (batch, 1, 28, 28).

    Pixel values come out between -1 and 1 because of the final Tanh.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Noise -> 7*7*128 numbers, later reshaped into 128 feature maps of 7x7
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        # 7x7 -> 14x14: (7 - 1)*2 - 2*1 + 4 = 14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # 14x14 -> 28x28: (14 - 1)*2 - 2*1 + 4 = 28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)                     # (batch, 6272)
        x = x.view(-1, 128, 7, 7)          # (batch, 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))  # (batch, 64, 14, 14)
        x = self.tanh(self.deconv2(x))     # (batch, 1, 28, 28)
        return x


class Discriminator(nn.Module):
    """Scores a 28x28 grayscale image (batch, 1, 28, 28) as real or fake: output (batch, 1).

    The output is a raw score (logit). Applying a sigmoid turns it into the
    probability that the image is real; during training, BCEWithLogitsLoss
    applies that sigmoid internally.
    """

    def __init__(self):
        super().__init__()
        # 28x28 -> 14x14: (28 + 2*1 - 4)/2 + 1 = 14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        # 14x14 -> 7x7: (14 + 2*1 - 4)/2 + 1 = 7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))             # (batch, 64, 14, 14)
        x = self.act2(self.bn2(self.conv2(x)))   # (batch, 128, 7, 7)
        x = self.flatten(x)                      # (batch, 6272)
        return self.fc(x)                        # (batch, 1)
