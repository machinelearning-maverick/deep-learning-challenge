import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.initializers import HeNormal, GlorotUniform

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def model_with_initialization(strategy):
    # Prepare data
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    # Define model
    model = Sequential(
        [
            Dense(
                10, activation="relu", input_shape=(4,), kernel_initializer=HeNormal()
            ),
            Dense(3, activation="softmax"),
        ]
    )

    # Compile
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Train
    history = model.fit(X, y, epochs=500, verbose=0)
    pass
