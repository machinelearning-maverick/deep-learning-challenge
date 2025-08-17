# Day 23: CNN with PyTorch (MINST -> CIFAR-10)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ====== CONFIG ======
DATASET = "mnist"  # MNIST or cifar10
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================

# Data transforms (normalize per dataset)
if DATASET.lower() == "mnist":
    in_channels = 1
    num_classes = 10

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean, std of MNIST
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(
        root="./data", train=True, transform=train_transforms, download=True
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, transform=test_transforms, download=True
    )

elif DATASET.lower() == "cifar10":
    in_channels = 3
    num_classes = 10

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                (0.2023, 0.1994, 0.2010),  # CIFAR-10 std
            ),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_ds = datasets.CIFAR10(
        root="./data", train=True, transform=train_transforms, download=True
    )
    test_ds = datasets.CIFAR10(
        root="./data", train=False, transform=test_transforms, download=True
    )
else:
    raise ValueError("DATASET must be 'mnist' or 'cifar10'.")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


# Model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 32 x H x W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 x H/2 x W/2
            #
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x H/2 x W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 x H/4 x W/4
            #
            # Unify spatial size to 7x7, that FC layer input is consistent
            # 64 x 7 x 7  (MNIST ends up same; CIFAR-10 adapts from 8x8)
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    pass


def evaluate(model, loader, criterion, device):
    pass
