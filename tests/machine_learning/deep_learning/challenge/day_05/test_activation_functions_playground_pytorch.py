from deep_learning_challenge.machine_learning.deep_learning.challenge.day_05.activation_functions_playground_pytorch import (
    get_activations_pytorch,
)


def test_get_activations_pytorch_shapes():
    x, activations = get_activations_pytorch()
    for name, y in activations.items():
        assert (
            y.shape == x.shape
        ), f"{name} output shape mismatch: {y.shape} != {x.shape}"
