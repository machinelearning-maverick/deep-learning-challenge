import torch

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_03.mlp_bin_class_hid_layer_pytorch import (
    train_mlp_on_xor,
)


def test_train_mlp_on_xor():
    # Dataset: XOR (nonlinear, binary classification)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    preds, _ = train_mlp_on_xor(X, y)
    expected = [0, 1, 1, 0]

    for i, (pred, exp) in enumerate(zip(preds, expected)):
        assert int(pred) == exp, f"Mismatch at index {i}: got {pred}, expected {exp}"
