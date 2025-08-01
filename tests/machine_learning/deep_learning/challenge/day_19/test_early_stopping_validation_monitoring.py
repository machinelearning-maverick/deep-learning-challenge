from deep_learning_challenge.machine_learning.deep_learning.challenge.day_19.early_stopping_validation_monitoring import (
    train_with_early_stopping,
    plot_train_vs_val_loss,
)


def test_train_and_plot():
    train_loss_history, val_loss_history = train_with_early_stopping()
    plot_train_vs_val_loss(train_loss_history, val_loss_history)
    pass
