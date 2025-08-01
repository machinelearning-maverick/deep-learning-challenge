import torch
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():
    # Data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=0,
        random_state=42,
    )

    # Train & Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTprch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test


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


def plot_train_vs_val_loss(train_loss_history, val_loss_history):
    fig, ax = plt.subplots()

    ax.plot(train_loss_history, label="Train Loss")
    ax.plot(val_loss_history, "Val Loss")
    ax.axvline(
        len(val_loss_history) - wait, color="red", linestyle="--", label="Early Stop"
    )
    ax.xlabel("Epoch")
    ax.ylabel("Loss")
    ax.title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True)
    ax.show()
