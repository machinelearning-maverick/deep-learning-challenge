from deep_learning_challenge.machine_learning.deep_learning.challenge.day_17.dropout_L2_regularization_pytorch import (
    prepare_data,
    prepare_train_data_loader,
    model_criterion_optimizer,
    train_model,
    evaluate_model,
    plot_loss_vs_accuracy,
)


def test_plot():
    X_train, X_test, y_train, y_test = prepare_data()
    loader = prepare_train_data_loader(X_train, y_train)
    model, criterion, optimizer = model_criterion_optimizer()
    loss_history, acc_history = train_model(model, loader, optimizer, criterion)

    evaluate_model(model, X_test, y_test)
    plot_loss_vs_accuracy(loss_history, acc_history)
