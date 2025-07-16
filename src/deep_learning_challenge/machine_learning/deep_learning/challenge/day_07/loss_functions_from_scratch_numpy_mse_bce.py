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


def gradients_bce_mse_plot():
    p = np.linspace(0.01, 0.99, 100)

    # Ground truth y = 1
    y_true = 1

    # Gradients
    grad_mse = 2 * (p - y_true)
    grad_bce = (p - y_true) / (p * (1 - p))

    plt.figure(figsize=(8, 5))
    plt.plot(p, grad_mse, label="∂MSE/∂ŷ (y=1)")
    plt.plot(p, grad_bce, label="∂BCE/∂ŷ (y=1)")
    plt.title("Gradient of MSE vs BCE when y=1")
    plt.xlabel("ŷ (Predicted Probability)")
    plt.ylabel("Gradient")
    plt.axhline(0, color="black", linestyle="--")
    plt.legend()

    plt.grid(True)
    plt.savefig("gradients_bce_mse_plot.png")
    plt.show()

    def manual_training_loop_using_erivatives():
        pass


if __name__ == "__main__":
    bce_mse_plot()
    gradients_bce_mse_plot()
