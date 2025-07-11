import torch
import torch.nn as nn
import torch.optim as optim

# XOR-like but linearly separable (AND gate)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

# Define model
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


# Prediction
with torch.no_grad():
    preds = model(X).round()
    print(f"Predictions: {preds.squeeze().numpy()}")
