import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def prepare_data(n_samples=500, n_features=4, n_classes=3, n_informative=3):
    # X.shape = (n_samples, n_input)
    # y.shape(500,) -> y has class indices -> requires 'one-hot' + 'softmax' + 'cross-entropy'
    X, y = make_classification(
        n_samples=n_samples,  # examples, rows in X
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,  # suggests complexity; used to tune 'n_hidden'
        n_redundant=0,
        random_state=42
    )

    # y_onehot.shape = (n_samples, n_output)
    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    return X, y, y_onehot


def backpropagation(X, y, y_onehot, n_features=4, n_classes=3):
    # Local model architecture parameters
    # Kept inside function for modular testing
    n_input = n_features
    n_output = n_classes  # for 'softmax' to output 3-class probabilities
    n_hidden = 10  # number of neurons in hidden layer - design choice

    lr = 0.01  # learning rate
    epochs = 500  # train the model for 500 full passes through all 'n_samples'

    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))

    W2 = np.random.randn(n_hidden, n_output)
    b2 = np.zeros((1, n_output))

    # Training loop
    losses = []

    for epoch in range(epochs):
        # Forward
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Loss
        loss = cross_entropy(y_onehot, A2)
        losses.append(loss)

        # Backward
        dZ2 = A2 - y_onehot
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # Print loss
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return losses, W1, b1, W2, b2


def loss_over_epochs_plot(losses):
    # Plot the loss curve
    plt.plot(losses)
    plt.title("Loss over epochs (manual backprop)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy loss")
    plt.grid()

    plt.savefig("loss_over_epoch-manual_backprop.png")
    plt.show()


def data_backprop_plot():
    n_features = 4
    n_classes = 3

    X, y, y_onehot = prepare_data(n_features=n_features, n_classes=n_classes)
    losses, W1, b1, W2, b2 = backpropagation(X, y, y_onehot, n_features=n_features, n_classes=n_classes)
    loss_over_epochs_plot(losses)
