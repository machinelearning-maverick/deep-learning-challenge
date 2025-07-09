import numpy as np

# Features: binary
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

# Labels: AND gate
y = np.array([0, 0, 0, 1])


def step(z):
    return 1 if z >= 0 else 0


def predict(x, w, b):
    return step(np.dot(x, w) + b)


def train_perceptron(X, y, lr=0.1, epochs=10):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        for i in range(n_samples):
            x_i = X[i]
            y_hat = predict(x_i, w, b)
            error = y[i] - y_hat

            w += lr * error * x_i
            b += lr * error
        print(f"Epoch {epoch}: weights = {w}, bias = {b}")

    return w, b


# Train/Fit
weights, bias = train_perceptron(X, y)

# Predict/Guess
preds = [predict(x, weights, bias) for x in X]
print(f"Guessings: {preds}")
