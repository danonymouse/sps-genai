"""Train the diffusion model on CIFAR-10 and save its weights for the API.

Usage (from the project root):
    uv run python -m training.train_diffusion --epochs 30

Follows Module 8 Practical 2: mix images with noise at a random noise level,
train the UNet to predict that noise, and keep a slowly-updated EMA copy of
the network for generating images.
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from app.diffusion_model import UNet, denoise, generate_images, offset_cosine_diffusion_schedule

WEIGHTS_PATH = Path("app/weights/cifar10_diffusion.pt")
SAMPLES_DIR = Path("training/diffusion_samples")

# The practical assumes images with mean 0 and variance 1, so we normalize each
# color channel with CIFAR-10's mean and standard deviation.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_noise(images):
    """Mix each image with Gaussian noise at a random diffusion time t in [0, 1]."""
    noises = torch.randn_like(images)
    times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
    noise_rates, signal_rates = offset_cosine_diffusion_schedule(times)
    noisy = signal_rates * images + noise_rates * noises
    return noisy, noises, noise_rates, signal_rates


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    network = UNet().to(device)
    ema_network = copy.deepcopy(network).eval()  # smoothed copy used for generating
    loss_fn = nn.L1Loss()  # mean absolute error between predicted and true noise
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)

    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    sample_gen = torch.Generator(device=device).manual_seed(0)  # same starting noise each time
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"UNet parameters: {sum(p.numel() for p in network.parameters()):,}")

    print(f"{'epoch':>5}  {'train loss':>10}  {'test loss':>9}")
    for epoch in range(1, args.epochs + 1):
        network.train()
        train_total = 0.0
        for images, _ in train_loader:  # class labels are not needed
            noisy, noises, noise_rates, signal_rates = add_noise(images.to(device))
            pred_noises, _ = denoise(network, noisy, noise_rates, signal_rates)
            loss = loss_fn(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()

            with torch.no_grad():  # EMA: slowly blend the new weights into the copy
                for ema_p, p in zip(ema_network.parameters(), network.parameters()):
                    ema_p.mul_(args.ema).add_(p, alpha=1 - args.ema)

        test_total = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                noisy, noises, noise_rates, signal_rates = add_noise(images.to(device))
                pred_noises, _ = denoise(ema_network, noisy, noise_rates, signal_rates)
                test_total += loss_fn(pred_noises, noises).item()

        print(f"{epoch:>5}  {train_total / len(train_loader):>10.4f}  {test_total / len(test_loader):>9.4f}")

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            sample_gen.manual_seed(0)
            samples = generate_images(ema_network, 64, diffusion_steps=20, mean=mean, std=std,
                                      generator=sample_gen, device=device)
            save_image(samples, SAMPLES_DIR / f"epoch_{epoch:03d}.png", nrow=8)

    # Save the EMA network (the one used for generating) and the normalization
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ema_model_state_dict": {k: v.cpu() for k, v in ema_network.state_dict().items()},
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    }, WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()