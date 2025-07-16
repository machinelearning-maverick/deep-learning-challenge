import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    """Mean Squared Error - Loss Function"""
    return np.mean((y_true - y_pred) ** 2)


def bce(y_true, y_pred, eps=1e-15):
    """Binary Cross-Entropy - Loss Function"""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
