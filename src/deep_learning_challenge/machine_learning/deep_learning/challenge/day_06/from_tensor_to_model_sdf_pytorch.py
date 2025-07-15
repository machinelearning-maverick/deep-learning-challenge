import torch
import torch.nn as nn

# Dummy input: batch of samples 4, 3 features each
x = torch.randn(4, 3)

# Define a simple Multi-Layer Perceptron
model = nn.Sequential(
    nn.Linear(3, 5),  # input: 3 features; output: 5 neurons
    nn.ReLU(),
    nn.Linear(5, 2),  # input: 5 neurons; output: 2 classes
)

# Forward pass
output = model(x)

# Inspect shapes
print(f"Input shape: {x.shape}")
for i, layer in enumerate(model):
    x = layer(x)
    print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")