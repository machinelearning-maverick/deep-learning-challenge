import tensorflow as tf


def bce_loss_keras(y_true, y_pred):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss = loss_fn(y_true, y_pred).numpy()

    print(f"Keras BinaryCrossentropy: {loss}")
    return loss
