import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_01.perceptron_numpy import (
    step,
    predict,
    train_perceptron,
)


def test_step_function():
    assert step(-1) == 0
    assert step(0) == 1
    assert step(0.5) == 1


def test_predict():
    x = np.array([1, 1])
    w = np.array([0.5, 0.5])

    b = -1
    assert predict(x, w, b) == 1

    b = -1.5
    assert predict(x, w, b) == 0


def test_train_on_and_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    w, b = train_perceptron(X, y, lr=0.1, epochs=10)
    preds = [predict(x, w, b) for x in X]

    assert preds == y.tolist()
