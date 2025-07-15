import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU


def keras_shape_debug(batch_size=4, input_dim=3):
    # Dummy input: batch of samples 4, 3 features each
    x = np.random.randn(batch_size, input_dim)

    # Define a simple Multi-Layer Perceptron
    model = Sequential([
        Dense(5, input_shape=(input_dim,), name="dense_1"), # input: 3 features; output: 5 neurons
        ReLU(name="relu_1"),
        Dense(2, name="dense_2"), # hidden; input: 5 neurons; output: 2 classes/units
    ])

    # Forward pass
    output = model.predict(x, verbose=0)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Inspect shapes; model summary
    model.summary()

    return [x.shape, (batch_size, 5), output.shape]
