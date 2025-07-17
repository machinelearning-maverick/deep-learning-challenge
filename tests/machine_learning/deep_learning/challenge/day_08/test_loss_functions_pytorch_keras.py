import torch

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.loss_functions_pytorch_keras import (
    bce_loss_pytorch,
)


def test_bce_loss_pytorch():
    # Simulated predictions (probabilities) and targets
    y_pred = torch.tensor([[0.9], [0.1]], dtype=torch.float32)
    y_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss = bce_loss_pytorch(y_pred, y_true)
    assert loss < 0.2
