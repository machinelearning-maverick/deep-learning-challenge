# Day 23: CNN with PyTorch (MINST -> CIFAR-10)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import dataset, transforms

# ====== CONFIG ======
DATASET = "minst"  # minst or cifar10
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================

# Data transforms (normalize per dataset)
if DATASET.lower() == "minst":
    pass
elif DATASET.lower() == "cifar10":
    pass
else:
    raise ValueError("DATASET must be 'minst' or 'cifar10'.")


# Model
class SimpleCNN(nn.Module):
    pass


def train_one_epoch(model, loader, optimizer, criterion, device):
    pass


def evaluate(model, loader, criterion, device):
    pass
