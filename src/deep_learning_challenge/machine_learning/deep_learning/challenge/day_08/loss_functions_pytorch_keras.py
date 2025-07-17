import torch
import torch.nn as nn


def bce_loss_pytorch(y_pred, y_true):

    loss_fn = nn.BCELoss()
    loss = loss_fn(y_pred, y_true)

    print(f"PyTorch BCE Loss: {loss.item()}")
    return loss
