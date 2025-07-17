import torch

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.loss_functions_pytorch import (
    bce_loss_pytorch,
    bce_logits_loss_pytorch,
    ce_logits_loss_pytorch,
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


def test_ce_logits_loss_pytorch():
    # logits shape: [batch, classes]
    y_logits = torch.tensor([[2.0, 0.5, 0.3], [0.2, 0.1, 2.1]])
    # y_true - true class indicies
    y_true = torch.tensor([0, 2])

    loss = ce_logits_loss_pytorch(y_logits, y_true)
    assert loss < 0.30
