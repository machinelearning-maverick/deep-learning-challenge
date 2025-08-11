import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_22.intro_to_cnn_by_hand import (
    conv2d_manual,
    create_plot,
    verification_with_pytorch,
)

# 1. Create a 5x5 grayscale "image"
IMAGE = np.array(
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
KERNEL = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)


def test_create_plot():
    feature_map = conv2d_manual(IMAGE, KERNEL, stride=1, padding=0)

    create_plot(IMAGE, KERNEL, feature_map)

    pass


def test_verification_with_pytorch():
    output = verification_with_pytorch(IMAGE, KERNEL)
    pass
