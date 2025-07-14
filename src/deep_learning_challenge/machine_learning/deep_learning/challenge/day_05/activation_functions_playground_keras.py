import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def get_activations_keras(x_vector = np.linspace(-3, 3, 200)):
    # Activation functions
    activations = {
        "ReLU": K.eval(K.relu(x_vector)),
        "Sigmoid": K.eval(K.sigmoid(x_vector)),
        "Tanh": K.eval(K.tanh(x_vector)),
        "LeakyReLU": K.eval(K.relu(x_vector, alpha=0.1)),  # aplha == negative slop
        "Softplus": K.eval(K.softplus(x_vector)),
    }

    return x_vector, {k: v for k, v in activations.items()}


def plot_activations_keras():
    x, activations = get_activations_keras()

    # Plot
    plt.figure(figsize=(10, 6))
    for name, y in activations.items():
        plt.plot(x, y, label=name)

    plt.plot(x, x, label="x (raw input)", linestyle="--", color="gray")

    plt.title("Activation Functions (Keras)")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plt.savefig("activation_functions_playground_keras.png")

    plt.show()
