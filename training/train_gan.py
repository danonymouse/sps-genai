"""Train the Assignment 3 GAN on MNIST and save the Generator's weights for the API.

Usage (from the project root):
    uv run python -m training.train_gan --epochs 30

This is a standard GAN, not the Wasserstein GAN from the class practical:
the Discriminator outputs a real/fake score and both networks are trained
with binary cross-entropy.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from app.gan_model import LATENT_DIM, Discriminator, Generator

WEIGHTS_PATH = Path("app/weights/gan_generator.pt")
SAMPLES_DIR = Path("training/gan_samples")

# Scale MNIST pixels from [0, 1] to [-1, 1] so real images match the
# range of the Generator's Tanh output.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train the Assignment 3 GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=TRANSFORM)
    # drop_last avoids a tiny final batch, which BatchNorm handles poorly
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy in one step
    # Adam with beta1 = 0.5 is the usual choice for stable GAN training
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # The same 64 noise vectors every epoch, so saved samples show progress
    fixed_noise = torch.randn(64, LATENT_DIM, device=device)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'epoch':>5}  {'D loss':>7}  {'G loss':>7}  {'D(real)':>7}  {'D(fake)':>7}")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        d_loss_sum = g_loss_sum = d_real_sum = d_fake_sum = 0.0

        for real, _ in loader:  # MNIST labels are not needed
            real = real.to(device)
            batch = real.size(0)
            real_labels = torch.ones(batch, 1, device=device)
            fake_labels = torch.zeros(batch, 1, device=device)

            # 1) Discriminator step: learn to score real images as 1 and fakes as 0
            noise = torch.randn(batch, LATENT_DIM, device=device)
            fake = generator(noise)
            real_scores = discriminator(real)
            fake_scores = discriminator(fake.detach())  # detach: don't update the Generator here
            d_loss = loss_fn(real_scores, real_labels) + loss_fn(fake_scores, fake_labels)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2) Generator step: learn to make the Discriminator score fakes as real
            g_loss = loss_fn(discriminator(fake), real_labels)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_real_sum += torch.sigmoid(real_scores).mean().item()
            d_fake_sum += torch.sigmoid(fake_scores).mean().item()

        n = len(loader)
        print(f"{epoch:>5}  {d_loss_sum / n:>7.4f}  {g_loss_sum / n:>7.4f}  "
              f"{d_real_sum / n:>7.3f}  {d_fake_sum / n:>7.3f}")

        # Save a grid of 64 generated digits on the first, every 5th, and last epoch
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_image(samples, SAMPLES_DIR / f"epoch_{epoch:03d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))

    # Only the Generator is needed to create new digits in the API
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: t.cpu() for name, t in generator.state_dict().items()}, WEIGHTS_PATH)
    print(f"Generator weights saved to {WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
