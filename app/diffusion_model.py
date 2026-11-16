"""Diffusion model for CIFAR-10 (Assignment 4), adapted from Module 8 Practical 2.

A UNet learns to predict the noise that was mixed into an image. To generate,
we start from pure noise and remove the predicted noise a little at a time
(reverse diffusion).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32  # CIFAR-10 (the practical used 64x64)
CHANNELS = 3


# ---------- Diffusion schedule ----------

def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Map a diffusion time t in [0, 1] to (noise_rate, signal_rate).

    noise_rate**2 + signal_rate**2 == 1, so mixing a unit-variance image with
    unit-variance noise keeps the variance at 1 (as derived in the practical).
    t = 0 is almost the clean image; t = 1 is almost pure noise.
    """
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(angles), torch.cos(angles)


# ---------- Building blocks (from the practical) ----------

class SinusoidalEmbedding(nn.Module):
    """Turns the noise variance (one number per image) into a 32-number vector
    of sines and cosines at 16 frequencies, so the network knows how noisy its
    input is."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x),
                          torch.cos(self.angular_speeds * x)], dim=-1)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus a shortcut that adds the input back in."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A 1x1 convolution matches channel counts when they differ
        self.proj = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.conv1(x))  # silu == swish: x * sigmoid(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks, saving each output as a skip connection, then halve the size."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    """Double the size, then residual blocks that each take in a saved skip connection."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """The practical's UNet: predicts the noise in a noisy image.

    Input:  noisy images (B, 3, 32, 32) and noise variances (B, 1, 1, 1)
    Output: predicted noise (B, 3, 32, 32)
    Resolution: 32 -> 16 -> 8 -> 4 (bottleneck) -> 8 -> 16 -> 32
    """

    def __init__(self, image_size=IMAGE_SIZE, num_channels=CHANNELS):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)  # 32 values
        # Image features (32) + noise embedding (32) = 64 channels into the first block
        self.down1 = DownBlock(32, block_depth=2, in_channels=64)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)
        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)
        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # start by predicting zero noise, as in the practical

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)                   # (B, 1, 1, 32)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2),      # (B, 32, 1, 1)
                                  size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)                          # (B, 64, 32, 32)
        x = self.down1(x, skips)   # -> 16x16
        x = self.down2(x, skips)   # -> 8x8
        x = self.down3(x, skips)   # -> 4x4 bottleneck
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)     # -> 8x8
        x = self.up2(x, skips)     # -> 16x16
        x = self.up3(x, skips)     # -> 32x32
        return self.final(x)


# ---------- Denoising and generation ----------

def denoise(network, noisy_images, noise_rates, signal_rates):
    """Predict the noise, then use it to estimate the clean image."""
    pred_noises = network(noisy_images, noise_rates ** 2)
    pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
    return pred_noises, pred_images


@torch.no_grad()
def reverse_diffusion(network, initial_noise, diffusion_steps):
    """Start from pure noise and step from t = 1 down to t = 0, as in the practical."""
    network.eval()
    step_size = 1.0 / diffusion_steps
    current = initial_noise
    for step in range(diffusion_steps):
        t = torch.full((current.shape[0], 1, 1, 1), 1.0 - step * step_size, device=current.device)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(t)
        pred_noises, pred_images = denoise(network, current, noise_rates, signal_rates)
        next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule(t - step_size)
        # Re-mix the predicted clean image with the predicted noise at the next, lower noise level
        current = next_signal_rates * pred_images + next_noise_rates * pred_noises
    return pred_images


def generate_images(network, num_images, diffusion_steps, mean, std, generator=None, device="cpu"):
    """Generate images and undo the normalization, returning pixel values in [0, 1]."""
    noise = torch.randn((num_images, network.num_channels, network.image_size, network.image_size),
                        generator=generator, device=device)
    images = reverse_diffusion(network, noise, diffusion_steps)
    return (images * std + mean).clamp(0.0, 1.0)