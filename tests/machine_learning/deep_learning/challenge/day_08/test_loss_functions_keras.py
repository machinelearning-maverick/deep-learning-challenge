import tensorflow as tf

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.loss_functions_keras import (
    bce_loss_keras,
)


def test_bce_loss_keras():
    y_true = tf.constant([[1.0], [0.0]])
    y_pred = tf.constant([[1.9], [0.1]])

    loss = bce_loss_keras(y_true, y_pred)
    assert loss < 0.06
