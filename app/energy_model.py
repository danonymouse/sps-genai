"""Energy-based model for CIFAR-10 (Assignment 4), adapted from Module 8 Practical 1.

The network gives each image one number, its "energy": low for images that look
like CIFAR-10, high for anything else. New images are made by starting from
random noise and repeatedly stepping downhill in energy (Langevin dynamics).
"""

import torch
import torch.nn as nn

IMAGE_SIZE = 32
CHANNELS = 3  # CIFAR-10 is RGB (the class practical used 1-channel MNIST)


def swish(x):
    """Swish activation from the practical: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """CNN from the practical: (batch, 3, 32, 32) image -> (batch, 1) energy.

    Four stride-2 convolutions shrink the image 32 -> 16 -> 8 -> 4 -> 2,
    then two linear layers reduce it to a single energy value.
    """

    def __init__(self, in_channels=CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 8 -> 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)           # 4 -> 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def random_images(n, device="cpu", generator=None):
    """Uniform random noise in [-1, 1], the starting point for sampling."""
    return torch.rand((n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
                      device=device, generator=generator) * 2 - 1


def langevin_sample(model, images, steps, step_size, noise_std, generator=None):
    """Langevin dynamics, as in the practical's generate_samples().

    Each step: add a little random noise, then move every pixel a small step
    in the direction that lowers the energy. The gradient is taken with respect
    to the IMAGE, not the model's weights, which stay fixed.
    """
    model.eval()  # the energy model has no Dropout/BatchNorm; this is just good practice
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn(images.shape, device=images.device,
                                generator=generator) * noise_std
            images = (images + noise).clamp(-1.0, 1.0)
        images.requires_grad_(True)

        energy = model(images)
        grads, = torch.autograd.grad(energy, images, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)  # gradient clipping keeps sampling stable
            images = (images - step_size * grads).clamp(-1.0, 1.0)
    return images.detach()