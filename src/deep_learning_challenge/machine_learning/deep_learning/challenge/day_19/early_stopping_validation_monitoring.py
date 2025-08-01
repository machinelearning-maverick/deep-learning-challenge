import torch


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
        # Training phase
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()  # reset gradients
            output = model(xb)  # forward pass
            loss = criterion(output, yb)
            loss.backward()  # backward pass
            optimizer.step()  # weight update
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validate phase
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
        val_loss_history.append(val_loss)

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_loss_history, val_loss_history
