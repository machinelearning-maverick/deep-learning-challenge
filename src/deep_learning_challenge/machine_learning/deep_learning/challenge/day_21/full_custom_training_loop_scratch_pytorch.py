from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(test_size=0.2):
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=0,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_val, y_train, y_val


def prepare_dataset_dataloader(X_train, X_val, y_train, y_val, batch_size=32):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_tensor_dataset = TensorDataset(X_train, y_train)
    train_data_loader = DataLoader(
        train_tensor_dataset, batch_size=batch_size, shuffle=True
    )

    val_tensor_dataset = TensorDataset(X_val, y_val)
    val_data_loader = DataLoader(val_tensor_dataset, batch_size=batch_size)

    return train_data_loader, val_data_loader
