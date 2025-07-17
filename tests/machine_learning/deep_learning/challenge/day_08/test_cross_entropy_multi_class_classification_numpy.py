import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_08.cross_entropy_multi_class_classification_numpy import (
    softmax,
    cross_entropy,
)


def test_softmax_cross_entropy():
    # Iris Flower Classification
    # Classify flowers into 3 species:
    # 0 = setosa, 1 = versicolor, 2 = virginica

    # one-hot encoded true labels (3 samples, 3 classes)
    y_true = np.array(
        [
            [1, 0, 0],  # setosa
            [0, 1, 0],  # versicolor
            [0, 0, 1],  # virginica
        ]
    )

    # Predicted logits (not probabilities yet)
    logits_good = np.array(
        [
            [5.0, 1.0, 0.1],
            [0.1, 4.0, 0.2],
            [0.2, 0.3, 5.0],
        ]
    )

    logits_bad = np.array(
        [
            [0.1, 4.0, 5.0],
            [3.0, 1.0, 0.2],
            [0.5, 2.0, 2.5],
        ]
    )

    # Convert logits to probabilities
    y_pred_good = None
    y_pred_bad = None
    pass
