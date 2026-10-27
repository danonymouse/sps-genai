"""Generate handwritten digits with the trained GAN Generator and return them as a PNG."""

import io
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from app.gan_model import LATENT_DIM, Generator

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "gan_generator.pt"


class DigitGenerator:
    """Loads the trained Generator once; `generate_png` draws new digits from random noise."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.generator = Generator()
        self.generator.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.generator.eval()  # BatchNorm uses its stored statistics, not the batch's

    @torch.no_grad()
    def generate_png(self, num_images=16, seed=None, scale=4):
        """Return a PNG (as bytes) showing `num_images` generated digits in a grid.

        The same seed always produces the same digits; no seed gives new ones each call.
        `scale` enlarges each 28x28 digit so the grid is easy to see.
        """
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        noise = torch.randn(num_images, LATENT_DIM, generator=rng)
        images = self.generator(noise)                 # (n, 1, 28, 28), values in [-1, 1]
        images = ((images + 1) / 2).clamp(0, 1)        # -> [0, 1] for display

        per_row = math.ceil(math.sqrt(num_images))     # roughly square grid
        grid = make_grid(images, nrow=per_row, padding=2)          # (3, H, W)
        pixels = (grid.permute(1, 2, 0) * 255).round().byte().numpy()  # (H, W, 3)

        picture = Image.fromarray(pixels)
        picture = picture.resize((picture.width * scale, picture.height * scale),
                                 Image.NEAREST)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        return buffer.getvalue()
