import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


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


model = MLPWithRegularization()

# CrossEntropyLoss includes softmax
criterion = nn.CrossEntropyLoss()

# L2 regularization is controlled via 'weight_decay' in the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
