import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

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

    # Train & Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=3)
    pass


def create_model():
    model = models.Sequential(
        [
            layers.Input(shape=(20,)),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(3, activation="softmax"),
        ]
    )
    return model
