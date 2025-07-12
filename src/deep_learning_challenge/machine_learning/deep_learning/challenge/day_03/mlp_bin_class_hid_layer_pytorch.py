import torch
import torch.nn as nn
import torch.optim as optim

# Dataset: XOR (nonlinear, binary classification)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Define MLP (Multi-Layer Perceptron) model
model = nn.Sequential(
    nn.Linear(2, 4),  # input layer -> hidden layer (2 - 4 neurons)
    nn.ReLU(),  # non-linear activation
    nn.Linear(4, 1),  # hidden layer -> output layer (4 - 1)
    nn.Sigmoid(),  # binary output
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(5000):
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
