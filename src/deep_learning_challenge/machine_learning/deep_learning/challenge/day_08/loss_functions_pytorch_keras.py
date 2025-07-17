import torch
import torch.nn as nn

# Simulated predictions (probabilities) and targets
y_pred = torch.tensor([[0.9], [0.1]], dtype=torch.float32)
y_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

loss_fn = nn.BCELoss()
loss = loss_fn(y_pred, y_true)

print(f"PyTorch BCE Loss: {loss.item()}")
