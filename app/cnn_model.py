"""CNN architecture for Assignment 2 (Module 4), plus the preprocessing it expects."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGE_SIZE = 64  # the spec's input size: 64x64 RGB

# CIFAR-10 label order: model output index -> class name
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Shared by training and the API, so uploaded images are prepared exactly
# like the training images. CIFAR-10 images are 32x32, so they are resized
# up to 64x64 to match the spec.
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # HxWxC, 0-255 -> CxHxW, 0.0-1.0
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
                         std=(0.2470, 0.2435, 0.2616)),   # and standard deviations
])


class AssignmentCNN(nn.Module):
    """CNN from the Assignment 2 specification.

    Input:  RGB images, shape (batch, 3, 64, 64)
    Output: raw scores (logits) for each class, shape (batch, num_classes)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # 3 x 64 x 64 -> 16 x 64 x 64 (padding=1 keeps the spatial size)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # Halves height and width; reused after both conv layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16 x 32 x 32 -> 32 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # After the second pool: 32 channels x 16 x 16 = 8192 features
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 32, 16, 16)
        x = torch.flatten(x, start_dim=1)     # -> (batch, 8192)
        x = F.relu(self.fc1(x))               # -> (batch, 100)
        return self.fc2(x)                    # -> (batch, 10) logits