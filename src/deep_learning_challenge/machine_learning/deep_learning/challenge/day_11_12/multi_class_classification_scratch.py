import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

## Step 1: Setup dataset
# Create synthetic 3-class classification data
X, y = make_classification(
    n_classes=3,
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# one-hot encoded labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))


## Step 2: Model initialization
n_samples, n_features = X.shape
n_classes = y_onehot.shape[1]

# Initialize weights and bias
W = np.random.randn(n_features, n_classes)
b = np.zeros((1, n_classes))


## Step 3: Softmax + cross-entropy
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


## Step 4: Training loop
lr = 0.1
epochs = 500

loss_history = []
acc_history = []

for epoch in range(epochs):
    z = np.dot(X, W) + b
    y_pred = softmax(z)

    # Track loss over time
    loss = cross_entropy(y_onehot, y_pred)
    acc = np.mean(np.argmax(y_pred, axis=1) == y)

    # Gradient
    grad_z = y_pred - y_onehot
    grad_W = np.dot(X.T, grad_z) / n_samples
    grad_b = np.mean(grad_z, axis=0, keepdims=True)

    # Update
    W -= lr * grad_W
    b -= lr * grad_b

    # Track loss over time
    loss_history.append(loss)
    acc_history.append(acc)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")


## Step 5: Test predictions (accuracy check)
y_pred_final = softmax(np.dot(X, W) + b)
y_pred_labels = np.argmax(y_pred_final, axis=1)
acc_final = np.mean(y_pred_labels == y)
print(f"Final test accuracy (NumPy model): {acc_final:.4f}")


## Step 6: Plot loss over time
plt.figure(figsize=(10, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Accuracy plot (with final value)
plt.subplot(1, 2, 2)
plt.plot(acc_history, label="Accuracy", color="green")
plt.title(f"Accuracy over Epochs (Final: {acc_final:.2%})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_over_epochs_v2.png")
plt.show()
