import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def conv2d_manual(image, kernel, stride=1, padding=0):
    # Padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")

    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape

    out_h = (img_h - kernel_h) // stride + 1
    out_w = (img_w - kernel_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Sliding window
    for y in range(0, out_h):
        for x in range(0, out_w):
            region = image[
                y * stride : y * stride + kernel_h, x * stride : x * stride + kernel_w
            ]
            output[y, x] = np.sum(region * kernel)

    return output


def conv2d_rgb_manual(image, kernel):
    # Assumes same HxW for all channels
    h, w, c = image.shape
    kc, kh, kw = kernel.shape
    assert c == kc, "Kernel and image channels must match"

    # Only valid conv (no padding)
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            # Extract the patch (3x3x3)
            patch = image[y : y + kh, x : x + kw, :]  # shape (3,3,3)
            # Multiply elementwise & sum over all channels
            output[y, x] = np.sum(patch * np.transpose(kernel, (1, 2, 0)))

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


def verify_conv2d_manual_with_pytorch(image, kernel):
    # [batch=1, channel=1, h, w]
    img_t = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    # [out_ch=1, in_ch=1, h, w]
    kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)

    conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight.copy_(kernel_t)

    output = conv(img_t)
    print(f"PyTorch conv output:\n {output.squeeze()}")
    return output


def verify_conv2d_rgb_manual_with_pytorch(image_rgb, kernel_rgb):
    # Convert to PyTorch format: [batch, channels, height, width]

    # (1,3,3,3)
    image_t = torch.tensor(image_rgb.transpose(2, 0, 1)).unsqueeze(0)
    # (out_channels=1, in_channels=3, h=3, w=3)
    kernel_t = torch.tensor(kernel_rgb).unsqueeze(0)

    conv = nn.Conv2d(3, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight.copy_(kernel_t)

    output = conv(image_t)
    print(f"Output feature map:\n {output}")
    return output
