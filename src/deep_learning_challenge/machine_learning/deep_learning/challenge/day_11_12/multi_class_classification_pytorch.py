import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create synthetic 3-class classification data
X, y = make_classification(
    n_classes=3,
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Model
model = nn.Sequential(nn.Linear(2, 3))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        pred_labels = torch.argmax(output, axis=1)
        acc = (pred_labels == y_train_tensor).float().mean().item()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
