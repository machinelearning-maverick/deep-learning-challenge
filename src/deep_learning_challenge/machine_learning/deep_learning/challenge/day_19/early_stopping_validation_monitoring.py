def train_with_early_stopping(
    model, train_loader, val_data, optimizer, criterion, epochs=50, patience=5
):
    X_val, y_val = val_data
    best_loss = float("inf")
    best_model_state = None
    wait = 0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        pass
    pass
