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
    pass
else:
    raise ValueError("DATASET must be 'mnist' or 'cifar10'.")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)


# Model
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential()

        self.classifier = nn.Sequential()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion, device):
    pass


def evaluate(model, loader, criterion, device):
    pass
