import torch
import torch.nn as nn
import torch.optim as optim


def train_mlp_on_xor(X, y, hidden_size=4, lr=0.1, epochs=5000):
    # Define MLP (Multi-Layer Perceptron) model
    model = nn.Sequential(
        nn.Linear(2, hidden_size),  # input layer -> hidden layer (2 - 4 neurons)
        nn.ReLU(),  # non-linear activation
        nn.Linear(hidden_size, 1),  # hidden layer -> output layer (4 - 1)
        nn.Sigmoid(),  # binary output
    )

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Predictions
    with torch.no_grad():
        preds = model(X).round()
        print(f"Predictions: {preds.squeeze().tolist()}")

    return preds, model
