import numpy as np
import matplotlib.pyplot as plt


class SGDOptimizer:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, W, b, dW, db):
        W -= self.lr * dW
        b -= self.lr * db
        return W, b


class MomentumOptimizer:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_W = None
        self.v_b = None

    def step(self, W, b, dW, db):
        # Initialize momentum terms if first step
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = np.zeros_like(b)

        # Update velocity
        self.v_W = self.beta * self.v_W - self.lr * dW
        self.v_b = self.beta * self.v_b - self.lr * db

        # Update parameters
        W += self.v_W
        b += self.v_b

        return W, b


def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entroyp(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# Code like in the Day 11-12 -> multi_class_classification_scratch.py
def train_model(X, y, y_onehot, optimizer, epochs=500):
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]

    # Initialize weights and biases
    W = np.random.randn(n_features, n_classes)
    b = np.zeros((1, n_classes))

    loss_history = []
    acc_history = []
    W_history = []
    b_history = []

    for epoch in range(epochs):
        z = np.dot(X, W) + b
        y_pred = softmax(z)

        loss = cross_entroyp(y_onehot, y_pred)
        acc = np.mean(np.argmax(y_pred, axis=1) == y)

        grad_z = y_pred - y_onehot
        grad_W = np.dot(X.T, grad_z) / n_samples
        grad_b = np.mean(grad_z, axis=0, keepdims=True)

        W, b = optimizer.step(W, b, grad_W, grad_b)

        loss_history.append(loss)
        acc_history.append(acc)

        W_history.append(W.copy())
        b_history.append(b.copy())

    W_history = np.array(W_history)  # shape: (epochs, in_features, out_classes)
    b_history = np.array(b_history)  # shape: (epochs, 1, out_classes)

    return loss_history, acc_history, W_history, b_history


def plot_impact_on_training(
    loss_history, acc_history, W_history, b_history, optimizer: str
):

    fig, ((loss_ax, acc_ax), (weights_ax, biases_ax)) = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8)
    )

    ## Loss plot
    loss_ax.plot(loss_history)

    loss_title = "Loss over epochs (training loop - scratch) \n"
    loss_title += f"Optimizer: {optimizer} \n"

    loss_ax.set_title(loss_title)
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    loss_ax.grid(True)

    ## Accuracy plot
    acc_ax.plot(acc_history)

    acc_title = "Accuracy over epochs (training loop - scratch) \n"
    acc_title += f"Optimizer: {optimizer} \n"

    acc_ax.set_title(acc_title)
    acc_ax.set(xlabel="Epoch", ylabel="Accuracy")
    acc_ax.grid(True)

    ## Weights plot
    # Weight evolution (each component over time)
    for i in range(W_history.shape[1]):        # input dim
        for j in range(W_history.shape[2]):    # output dim
            weights_ax.plot(W_history[:, i, j], label=f"W[{i},{j}]")

    weights_title = "Weights (training loop - scratch) \n"
    weights_title += f"Optimizer: {optimizer} \n"

    weights_ax.set_title(weights_title)
    weights_ax.set(xlabel="Epoch", ylabel="Weight")
    weights_ax.grid(True)

    ## Biases plot
    # Bias evolution
    for j in range(b_history.shape[2]):
        biases_ax.plot(b_history[:, 0, j], label=f"b[{j}]")

    biases_title = "Biases (training loop - scratch) \n"
    biases_title += f"Optimizer: {optimizer} \n"

    biases_ax.set_title(biases_title)
    biases_ax.set(xlabel="Epoch", ylabel="Bias")
    biases_ax.grid(True)

    plt.tight_layout()

    plt.savefig(f"impact-on-training_{optimizer}-optimizer-scratch.png")
    plt.show()
