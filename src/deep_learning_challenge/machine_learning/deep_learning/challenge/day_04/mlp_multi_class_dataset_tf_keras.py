import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_iris_mlp(hidden_units=10, epochs=100):
    # Load & prepare Iris dataset
    iris_data = load_iris()
    X = iris_data.data  # shape (150, 4)
    y = iris_data.target.reshape(-1, 1)  # shape (150,) -> (150, 1)

    # One-hot encode labels: [0, 1, 2] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    # Standarize features: (across column) has mean==0 & std==1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, train_size=0.2, random_state=42
    )

    # Build a simple MLP model
    model = Sequential(
        [
            Dense(hidden_units, input_shape=(4,), activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model Loss: {loss:.4f}")

    return accuracy
