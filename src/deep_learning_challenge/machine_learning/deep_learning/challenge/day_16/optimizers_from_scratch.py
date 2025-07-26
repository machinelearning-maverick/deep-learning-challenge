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
