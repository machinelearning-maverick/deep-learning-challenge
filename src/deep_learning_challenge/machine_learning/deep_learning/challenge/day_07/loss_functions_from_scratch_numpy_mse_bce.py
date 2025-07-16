import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
