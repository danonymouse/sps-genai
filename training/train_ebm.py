"""Train the energy-based model on CIFAR-10 and save its weights for the API.

Usage (from the project root):
    uv run python -m training.train_ebm --epochs 20

Follows Module 8 Practical 1: contrastive divergence loss, Langevin dynamics
sampling, and a replay buffer of past samples.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from app.energy_model import EnergyModel, langevin_sample, random_images

WEIGHTS_PATH = Path("app/weights/cifar10_ebm.pt")
SAMPLES_DIR = Path("training/ebm_samples")

# Scale pixels from [0, 1] to [-1, 1], the range the sampler works in
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class Buffer:
    """Replay buffer from the practical.

    Langevin sampling needs many steps to turn noise into a good image, so
    instead of starting from scratch each batch, we keep past samples and keep
    refining them. About 5% of each batch starts fresh from random noise.
    """

    def __init__(self, model, device, batch_size, max_size=8192):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_size = max_size
        self.examples = list(torch.split(random_images(batch_size, device), 1, dim=0))

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch_size, 0.05)
        new_imgs = random_images(n_new, self.device)
        old_imgs = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)
        imgs = langevin_sample(self.model, torch.cat([new_imgs, old_imgs], dim=0),
                               steps, step_size, noise)
        self.examples = (list(torch.split(imgs, 1, dim=0)) + self.examples)[:self.max_size]
        return imgs


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_step(model, buffer, real_imgs, optimizer, args):
    """One update: lower the energy of real images, raise it for generated ones."""
    real_imgs = (real_imgs + torch.randn_like(real_imgs) * args.noise).clamp(-1.0, 1.0)
    fake_imgs = buffer.sample_new_exmps(args.steps, args.step_size, args.noise)

    model.train()
    out = model(torch.cat([real_imgs, fake_imgs], dim=0))
    real_out, fake_out = torch.split(out, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

    cdiv_loss = real_out.mean() - fake_out.mean()  # contrastive divergence
    reg_loss = args.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())  # keeps energies small
    loss = cdiv_loss + reg_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    return loss.item(), cdiv_loss.item(), real_out.mean().item(), fake_out.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train an energy-based model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1, help="regularization strength")
    parser.add_argument("--steps", type=int, default=60, help="Langevin steps per batch")
    parser.add_argument("--step-size", type=float, default=10.0)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = Buffer(model, device, args.batch_size)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'epoch':>5}  {'loss':>8}  {'cdiv':>8}  {'E(real)':>8}  {'E(fake)':>8}")
    for epoch in range(1, args.epochs + 1):
        totals = np.zeros(4)
        for real, _ in loader:  # class labels are not needed
            totals += train_step(model, buffer, real.to(device), optimizer, args)
        loss, cdiv, e_real, e_fake = totals / len(loader)
        print(f"{epoch:>5}  {loss:>8.4f}  {cdiv:>8.4f}  {e_real:>8.4f}  {e_fake:>8.4f}")

        # Save 64 of the newest buffer samples on the first, every 5th and last epoch
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            grid = torch.cat(buffer.examples[:64], dim=0)
            save_image(grid, SAMPLES_DIR / f"buffer_epoch_{epoch:03d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))

    # What the API does: start from pure noise and run many Langevin steps
    start = random_images(64, device)
    from_noise = langevin_sample(model, start, steps=500, step_size=args.step_size, noise_std=args.noise)
    save_image(from_noise, SAMPLES_DIR / "from_noise_500_steps.png",
               nrow=8, normalize=True, value_range=(-1, 1))

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: t.cpu() for name, t in model.state_dict().items()}, WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()