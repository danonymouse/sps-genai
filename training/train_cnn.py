
"""Train the Assignment 2 CNN on CIFAR-10 and save its weights for the API.

Usage (from the project root):
    uv run python -m training.train_cnn --epochs 10

Test accuracy is measured after every epoch, not just at the end, so the
log shows how well the model generalizes as training goes on.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import PREPROCESS, AssignmentCNN

WEIGHTS_PATH = Path("app/weights/cifar10_cnn.pt")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(batch_size):
    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=PREPROCESS)
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=PREPROCESS)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training set. Returns (average loss, accuracy)."""
    model.train()
    total_loss, n_correct, n_seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_seen += labels.size(0)
    return total_loss / n_seen, n_correct / n_seen


@torch.no_grad()
def evaluate(model, loader, device):
    """Accuracy on a dataset, with the model in eval mode and no gradients."""
    model.eval()
    n_correct, n_seen = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        n_correct += (model(images).argmax(dim=1) == labels).sum().item()
        n_seen += labels.size(0)
    return n_correct / n_seen


def main():
    parser = argparse.ArgumentParser(description="Train the Assignment 2 CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, test_loader = make_loaders(args.batch_size)
    model = AssignmentCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"{'epoch':>5}  {'loss':>7}  {'train acc':>9}  {'test acc':>8}")
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"{epoch:>5}  {loss:>7.4f}  {train_acc:>9.1%}  {test_acc:>8.1%}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: t.cpu() for name, t in model.state_dict().items()}, WEIGHTS_PATH)
    print(f"Final test accuracy: {test_acc:.2%}")
    print(f"Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()