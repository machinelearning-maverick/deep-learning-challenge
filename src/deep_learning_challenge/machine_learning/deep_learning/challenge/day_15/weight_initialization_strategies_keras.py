import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.initializers import HeNormal, GlorotUniform

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def model_with_initialization(X, y, strategy_hidden_layer, strategy_output_layer):
    # Define model
    model = Sequential(
        [
            Dense(
                10,
                activation="relu",
                input_shape=(4,),
                kernel_initializer=strategy_hidden_layer,
            ),
            Dense(
                3,
                activation="softmax",
                kernel_initializer=strategy_output_layer,
            ),
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
    loss, accuracy = model.evaluate(X, y, verbose=0)

    return history, loss, accuracy


def plot_impact_on_training(models_init_data):

    columns = len(models_init_data)
    fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(5 * columns, 5))

    for idx, ax in enumerate(fig.axes):
        ax.plot(models_init_data[idx]["history"].history["loss"])

        title = "Loss over epochs (Keras) \n"
        title += f"Initializers: {models_init_data[idx]["initializers"]} \n"
        title += f"Final loss: {models_init_data[idx]["loss"]} \n"
        title += f"Final acc: {models_init_data[idx]["accuracy"]} \n"

        ax.set_title(title)
        ax.set(xlabel="Epoch", ylabel="Loss")

    plt.tight_layout()

    plt.savefig("impact-on-training_Keras-Autograd.png")
    plt.show()
