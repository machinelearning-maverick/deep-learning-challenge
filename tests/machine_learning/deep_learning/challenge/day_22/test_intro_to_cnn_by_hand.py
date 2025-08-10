import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_22.intro_to_cnn_by_hand import (
    conv2d_manual,
    create_plot,
)


def test_create_plot():
    # 1. Create a 5x5 grayscale "image"
    image = np.array(
        [
            [10, 10, 10, 0, 0],
            [10, 10, 10, 0, 0],
            [10, 10, 10, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    # 2. Define a 3x3 edge detection kernel
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)

    feature_map = conv2d_manual(image, kernel, stride=1, padding=0)

    create_plot(image, kernel, feature_map)

    pass
