import tensorflow as tf

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_20.adam_rmsprop_batchnorm_train_loop_keras import (
    prepare_data,
    prepare_datasets,
    create_model,
    training_loop,
    plot_train_loss_vs_val_acc,
)


def test_training_loop_optimizer_adam():
    X_train, X_val, y_train_oh, y_val_oh = prepare_data()
    train_dataset, val_dataset = prepare_datasets(X_train, X_val, y_train_oh, y_val_oh)
    model = create_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    train_acc, val_acc, train_loss_history, val_loss_history, val_acc_history = (
        training_loop(model, optimizer, train_dataset, val_dataset)
    )

    plot_train_loss_vs_val_acc(
        train_loss_history, val_loss_history, val_acc_history, "Adam"
    )
    pass


def test_training_loop_optimizer_rmsprop():
    X_train, X_val, y_train_oh, y_val_oh = prepare_data()
    train_dataset, val_dataset = prepare_datasets(X_train, X_val, y_train_oh, y_val_oh)
    model = create_model()

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

    train_acc, val_acc, train_loss_history, val_loss_history, val_acc_history = (
        training_loop(model, optimizer, train_dataset, val_dataset)
    )

    plot_train_loss_vs_val_acc(
        train_loss_history, val_loss_history, val_acc_history, "RMSprop"
    )
    pass
