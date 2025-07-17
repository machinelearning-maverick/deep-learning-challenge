import tensorflow as tf

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.loss_functions_keras import (
    bce_loss_keras,
    cce_loss_keras,
)


def test_bce_loss_keras():
    y_true = tf.constant([[1.0], [0.0]])
    y_pred = tf.constant([[1.9], [0.1]])

    loss = bce_loss_keras(y_true, y_pred)
    assert loss < 0.06


def test_cce_loss_keras():
    y_true = tf.constant([[0, 0, 1], [1, 0, 0]])  # one-hot encoded
    y_pred = tf.constant([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])

    loss = cce_loss_keras(y_true, y_pred)
    assert loss < 0.2
