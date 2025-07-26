import numpy as np


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

    return loss_history, acc_history, W, b
