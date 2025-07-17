import tensorflow as tf


def bce_loss_keras(y_true, y_pred):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(y_true, y_pred).numpy()

    print(f"Keras BinaryCrossentropy: {loss}")
    return loss


def cce_loss_keras(y_true, y_pred):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_fn(y_true, y_pred).numpy()

    print(f"Keras CategoricalCrossentropy: {loss}")
    return loss


def scce_loss_keras(y_true, y_pred):
    """Loss function with class indices"""
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = loss_fn(y_true, y_pred).numpy()

    print(f"Keras SparseCategoricalCrossentropy: {loss}")
    return loss
