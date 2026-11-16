"""Generate CIFAR-10-style images with the trained energy and diffusion models (Assignment 4)."""

import io
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from app.diffusion_model import UNet, generate_images
from app.energy_model import EnergyModel, langevin_sample, random_images

WEIGHTS_DIR = Path(__file__).parent / "weights"


def to_png(images, scale=4):
    """Turn a batch of images with pixel values in [0, 1] into one PNG grid (as bytes).

    `scale` enlarges each 32x32 image so the grid is easy to see.
    """
    per_row = math.ceil(math.sqrt(images.shape[0]))  # roughly square grid
    grid = make_grid(images, nrow=per_row, padding=2)
    pixels = (grid.permute(1, 2, 0) * 255).round().byte().numpy()
    picture = Image.fromarray(pixels)
    picture = picture.resize((picture.width * scale, picture.height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def make_rng(seed):
    """Same seed -> same images; no seed -> different images every call."""
    return torch.Generator().manual_seed(seed) if seed is not None else None


class EnergyImageGenerator:
    """Loads the trained energy model once and samples images with Langevin dynamics."""

    STEP_SIZE = 10.0     # same values used in training
    NOISE_STD = 0.005

    def __init__(self, weights_path=WEIGHTS_DIR / "cifar10_ebm.pt"):
        self.model = EnergyModel()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def generate_png(self, num_images=16, steps=500, seed=None):
        rng = make_rng(seed)
        start = random_images(num_images, generator=rng)  # pure noise in [-1, 1]
        images = langevin_sample(self.model, start, steps, self.STEP_SIZE, self.NOISE_STD,
                                 generator=rng)
        return to_png((images + 1) / 2)  # [-1, 1] -> [0, 1]


class DiffusionImageGenerator:
    """Loads the trained diffusion UNet (EMA weights) once and generates images by reverse diffusion."""

    def __init__(self, weights_path=WEIGHTS_DIR / "cifar10_diffusion.pt"):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.network = UNet()
        self.network.load_state_dict(checkpoint["ema_model_state_dict"])
        self.network.eval()
        # Undo the per-channel normalization used in training
        self.mean = torch.tensor(checkpoint["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(checkpoint["std"]).view(1, 3, 1, 1)

    def generate_png(self, num_images=16, steps=50, seed=None):
        images = generate_images(self.network, num_images, steps, self.mean, self.std,
                                 generator=make_rng(seed))
        return to_png(images)  # already in [0, 1]