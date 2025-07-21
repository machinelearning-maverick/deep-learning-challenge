import numpy as np

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


def prepare_data():
    X, y = make_classification(
        n_samples=500, n_features=4, n_classes=3, n_informative=3
    )
    y_onehot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

    return X, y, y_onehot


def backpropagation(X, y, y_onehot):
    # Parameters, inside function for "clarity"
    n_input = 4
    n_hidden = 10
    n_output = 3

    lr = 0.01
    epochs = 500

    pass
