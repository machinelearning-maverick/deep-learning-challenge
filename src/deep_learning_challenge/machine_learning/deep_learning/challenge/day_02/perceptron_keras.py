import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Model
model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))

# Compile
model.compile(optimizer="sgd", loss="binary_crossentropy")

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict
preds = model.predict(X)
print(f"Predictions: {np.round(preds).flatten()}")