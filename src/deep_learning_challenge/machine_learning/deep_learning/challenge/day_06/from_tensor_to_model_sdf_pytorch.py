import torch
import torch.nn as nn


def pytorch_shape_debug(batch_size=4, input_dim=3):
    # Dummy input: batch of samples 4, 3 features each
    x = torch.randn(batch_size, input_dim)

    # Define a simple Multi-Layer Perceptron
    model = nn.Sequential(
        nn.Linear(input_dim, 5),  # input: 3 features; output: 5 neurons
        nn.ReLU(),
        nn.Linear(5, 2),  # input: 5 neurons; output: 2 classes
    )

    # Forward pass
    output = model(x)

    # Inspect shapes
    shapes = []
    shapes.append(x.shape)

    print(f"Input shape: {shapes}")
    for i, layer in enumerate(model):
        x = layer(x)
        shapes.append(x.shape)
        print(f"After layer {i} ({layer.__class__.__name__}): {shapes}")

    return shapes
