import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.utils import plot_model

# Dummy input: batch of samples 4, 3 features each
x = np.random.randn(4, 3)

# Define a simple Multi-Layer Perceptron
model = Sequential([
    Dense(5, input_shape=(3,), name="dense_1"), # input: 3 features; output: 5 neurons
    ReLU(name="relu_1"),
    Dense(2, name="dense_2"), # hidden; input: 5 neurons; output: 2 classes/units
])

# Forward pass
output = model.predict(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Inspect shapes; model summary
model.summary()
