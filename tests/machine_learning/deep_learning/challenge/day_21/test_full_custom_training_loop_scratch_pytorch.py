from deep_learning_challenge.machine_learning.deep_learning.challenge.day_21.full_custom_training_loop_scratch_pytorch import (
    prepare_data,
    prepare_dataset_dataloader,
    MultiLayerPerceptron,
    training_loop,
    plot_train_loss_vs_val_acc,
)


def test_training_loop_with_plotting():
    X_train, X_val, y_train, y_val = prepare_data()
    train_data_loader, val_data_loader = prepare_dataset_dataloader(
        X_train, X_val, y_train, y_val
    )

    model = MultiLayerPerceptron()
    train_loss, val_loss, val_acc = training_loop(
        model, train_data_loader, val_data_loader
    )

    plot_train_loss_vs_val_acc(train_loss, val_loss, val_acc)
    pass
