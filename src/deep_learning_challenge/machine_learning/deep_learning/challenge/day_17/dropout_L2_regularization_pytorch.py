import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():
    # Data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=0,
        random_state=42,
    )

    # Train & Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTprch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def prepare_train_data_loader(X_train, y_train):
    train_tensor_ds = TensorDataset(X_train, y_train)
    train_data_loader = DataLoader(train_tensor_ds, batch_size=32, shuffle=True)

    return train_data_loader


class MLPWithRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, loader, optimizer, criterion, epochs=50):
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in loader:
            optimizer.zero_grad()  # reset gradients
            out = model(xb)  # forward pass
            loss = criterion(out, yb)
            loss.backward()  # computes gradient
            optimizer.step()  # applies updates to weights
            total_loss += loss.item()

            # accuracy calculation
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    # avg loss & accuracy per epoch
    avg_loss = total_loss / len(loader)
    acc = correct / total

    print(f"Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}")

    return loss_history, acc_history


def model_criterion_optimizer():
    model = MLPWithRegularization()

    # CrossEntropyLoss includes softmax
    criterion = nn.CrossEntropyLoss()

    # L2 regularization is controlled via 'weight_decay' in the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    return model, criterion, optimizer


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
    print(f"Test Accuracy: {acc:.4f}")


def plot_loss_vs_accuracy(loss_history, acc_history):
    fig, (loss_ax, acc_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))

    loss_ax.plot(loss_history)

    acc_ax.plot(acc_history)



