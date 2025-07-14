import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Input values
x = torch.linspace(-5, 5, 200)

# Activation functions
activations = {
    "ReLU": F.relu(x),
    "Sigmoid": torch.sigmoid(x),
    "Tanh": torch.tanh(x),
    "LeakyReLU": F.leaky_relu(x, negative_slope=0.1),
    "Softplus": F.softplus(x),
    "Softmax": F.softmax(x),
}

# Plot
plt.figure(figsize=(10, 6))
for name, y in activations.items():
    plt.plot(x.numpy(), y.numpy(), label=name)

plt.title("Activation Functions (PyTorch)")
plt.xlabel("x")
plt.ylabel("activation(x)")
plt.legend()
plt.grid(True)

plt.savefig("activation_functions_playground.png")

plt.show()
