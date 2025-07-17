import torch
import torch.nn as nn


def bce_loss_pytorch(y_pred, y_true):
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_pred, y_true)

    print(f"PyTorch BCE Loss: {loss.item()}")
    return loss


def bce_logits_loss_pytorch(logits, y_true):
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, y_true)

    print(f"PyTorch BCEWithLogits Loss: {loss.item()}")
    return loss
