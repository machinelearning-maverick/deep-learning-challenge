import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def backprop_autograd_pytorch():
    # Prepare data
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    scaler = StandardScaler()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # needed for CrossEntropyLoss

    # Define model
    model = nn.Sequential(
        nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 3)
    )  # output logits

    # Loss & optimizer
    loss_fn = nn.CrossEntropyLoss()  # includes softmax internally
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()  # reset gradients
        logits = model(X)  # forward pass
        loss = loss_fn(logits, y)
        loss.backward()  # backward pass
        optimizer.step()  # weight update
        losses.append(loss.item())

    plt.plot(losses)
    plt.title("Loss over epochs (PyTorch Autograd)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.savefig("loss-epoch_PyTorch-Autograd.png")
    plt.show()


if __name__ == "__main__":
    backprop_autograd_pytorch()
