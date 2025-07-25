import torch
import torch.nn as nn
import torch.nn.init as init


def model_with_initialization(strategy):
    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )