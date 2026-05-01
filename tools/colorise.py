# tools/colorise.py

import cv2
import numpy as np


def get_pink_shade():
    """Generate a random shade of pink."""
    base = np.array([255, 105, 180])
    noise = np.random.randint(-40, 40, 3)
    return np.clip(base + noise, 0, 255)


def get_blue_shade():
    """Generate a random shade of blue."""
    base = np.array([65, 105, 225])
    noise = np.random.randint(-40, 40, 3)
    return np.clip(base + noise, 0, 255)


def colorize(mask: np.ndarray, color_mode: str = "random") -> np.ndarray:
    """
    Color each connected component (instance) in a binary mask.

    Args:
        mask (np.ndarray): Binary mask (0/255)
        color_mode (str): "random", "pink", or "blue"

    Returns:
        np.ndarray: Colored image (H, W, 3)
    """

    if mask is None:
        raise ValueError("Mask is None")

    # Ensure binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(mask)

    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    np.random.seed(42)

    for i in range(1, num_labels):  # skip background

        if color_mode == "pink":
            color = get_pink_shade()

        elif color_mode == "blue":
            color = get_blue_shade()

        else:
            color = np.random.randint(0, 255, 3)

        colored[labels == i] = color

    return colored