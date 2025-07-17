import torch

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.loss_functions_pytorch_keras import (
    bce_loss_pytorch,
    bce_logits_loss_pytorch,
)


def test_bce_loss_pytorch():
    # Simulated predictions (probabilities) and targets
    y_pred = torch.tensor([[0.9], [0.1]], dtype=torch.float32)
    y_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss = bce_loss_pytorch(y_pred, y_true)
    assert loss < 0.2


def test_bce_logits_loss_pytorch():
    # Simulated predictions (probabilities) and targets
    y_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    # raw outputs before applying activation function
    logits = torch.tensor([[2.0], [-2.0]], dtype=torch.float32)

    loss = bce_logits_loss_pytorch(logits, y_true)
    assert loss < 0.15
