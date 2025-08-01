from deep_learning_challenge.machine_learning.deep_learning.challenge.day_19.early_stopping_validation_monitoring import (
    prepare_data,
    prepare_train_data_loader,
    prepare_val_data,
    prepare_model,
    prepare_criterion_and_optimizer,
    train_with_early_stopping,
    plot_train_vs_val_loss,
)


def test_train_and_plot():
    X_train, X_val, y_train, y_val = prepare_data()
    train_loader = prepare_train_data_loader(X_train, y_train)
    val_data = prepare_val_data(X_val, y_val)
    model = prepare_model()
    criterion, optimizer = prepare_criterion_and_optimizer(model)

    train_loss, val_loss, stop_epoch, wait = train_with_early_stopping(
        model, train_loader, val_data, optimizer, criterion
    )

    plot_train_vs_val_loss(train_loss, val_loss, wait)

    pass
