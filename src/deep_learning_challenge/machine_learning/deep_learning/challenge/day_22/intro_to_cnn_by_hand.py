import numpy as np
import matplotlib.pyplot as plt


def conv2d_manual(img, kernel, stride=1, padding=0):
    # Padding
    if padding > 0:
        img = np.pad(img, ((padding, padding), (padding, padding)), mode="constant")

    kernel_h, kernel_w = kernel.shape
    img_h, img_w = img.shape

    out_h = (img_h - kernel_h) // stride + 1
    out_w = (img_w - kernel_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Sliding window
    for y in range(0, out_h):
        for x in range(0, out_w):
            region = img[
                y * stride : y * stride + kernel_h, x * stride : x * stride + kernel_w
            ]
            output[y, x] = np.sum(region * kernel)

    return output


def create_plot(image, kernel, feature_map):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Kernel")
    plt.imshow(kernel, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Feature Map")
    plt.imshow(feature_map, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
