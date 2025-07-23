import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def backprop_autograd_keras():
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
        [Dense(10, activation="relu", input_shape=(4,)), Dense(3, activation="softmax")]
    )

    # Compile
    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss=SparseCategoricalCrossentropy(),  # y - must be class index, not one-hot
        metrics=["accuracy"],
    )

    # Train
    history = model.fit(X, y, epochs=500, verbose=0)

    plt.plot(history.history["loss"])
    plt.title("Loss over epochs (Keras Autograd)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.savefig("loss-epoch_Keras-Autograd.png")
    plt.show()


if __name__ == "__main__":
    backprop_autograd_keras()
