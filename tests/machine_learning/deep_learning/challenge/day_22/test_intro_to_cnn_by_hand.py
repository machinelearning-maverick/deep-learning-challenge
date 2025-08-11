import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_22.intro_to_cnn_by_hand import (
    conv2d_manual,
    create_plot,
    verify_conv2d_manual_with_pytorch,
    verify_conv2d_rgb_manual_with_pytorch,
)

# A 5x5 grayscale "image"
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

# A 3x3 edge detection kernel
KERNEL = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)

# Tiny 3×3 RGB image
IMAGE_RGB = np.array(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # Row 1
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],  # Row 2
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],  # Row 3
    ],
    dtype=np.float32,
)  # Shape: (3,3,3)

print("Image shape:", IMAGE_RGB.shape)  # (height, width, channels)

# 3×3×3 kernel (one 3×3 filter per channel)
KERNEL_RGB = np.array(
    [
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],  # Red channel filter
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # Green channel filter
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],  # Blue channel filter
    ],
    dtype=np.float32,
)  # Shape: (channels, kernel_h, kernel_w)


def test_create_plot():
    feature_map = conv2d_manual(IMAGE, KERNEL, stride=1, padding=0)
    create_plot(IMAGE, KERNEL, feature_map)
    pass


def test_verify_conv2d_manual_with_pytorch():
    output = verify_conv2d_manual_with_pytorch(IMAGE, KERNEL)
    pass


def test_verify_conv2d_manual_with_pytorch():
    output = verify_conv2d_rgb_manual_with_pytorch(IMAGE_RGB, KERNEL_RGB)
    pass
