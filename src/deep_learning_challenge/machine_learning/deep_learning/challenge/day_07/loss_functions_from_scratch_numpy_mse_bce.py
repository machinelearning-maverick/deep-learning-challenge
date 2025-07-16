import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    """Mean Squared Error - Loss Function"""
    return np.mean((y_true - y_pred) ** 2)


def bce(y_true, y_pred, eps=1e-15):
    """Binary Cross-Entropy - Loss Function"""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_mse_plot():
    p = np.linspace(0.01, 0.99, 100)

    # For y = 1
    bce_1 = -np.log(p)
    mse_1 = (1 - p) ** 2

    # For y = 0
    bce_0 = -np.log(1 - p)
    mse_0 = (0 - p) ** 2

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(p, mse_1, label="MSE (y=1)")
    plt.plot(p, bce_1, label="BCE (y=1)")
    plt.title("Loss when y=1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(p, mse_0, label="MSE (y=0)")
    plt.plot(p, bce_0, label="BCE (y=0)")
    plt.title("Loss when y=0")
    plt.xlabel("Predicted Probability")
    plt.legend()

    plt.tight_layout()
    plt.savefig("bce_mse_plot.png")
    plt.show()


if __name__ == "__main__":
    bce_mse_plot()
