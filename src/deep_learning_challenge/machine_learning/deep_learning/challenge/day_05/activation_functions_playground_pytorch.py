import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


def get_activations_pytorch(x_vector = torch.linspace(-3, 3, 200)):
    # Activation functions
    activations = {
        "ReLU": F.relu(x_vector),
        "Sigmoid": torch.sigmoid(x_vector),
        "Tanh": torch.tanh(x_vector),
        "LeakyReLU": F.leaky_relu(x_vector, negative_slope=0.1),
        "Softplus": F.softplus(x_vector),
        "Softmax": F.softmax(x_vector),
    }

    return x_vector.numpy(), {k: v.numpy() for k, v in activations.items()}


def plot_activations_pytorch():
    x, activations = get_activations_pytorch()

    # Plot
    plt.figure(figsize=(10, 6))
    for name, y in activations.items():
        plt.plot(x, y, label=name)

    plt.plot(x, x, label="x (raw input)", linestyle="--", color="gray")

    plt.title("Activation Functions (PyTorch)")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.legend()
    plt.grid(True)

    # plt.savefig("activation_functions_playground_pytorch.png")

    plt.show()
