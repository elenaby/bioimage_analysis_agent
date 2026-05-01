# tools/segmentation.py

import cv2
import numpy as np


def segment(image: np.ndarray) -> np.ndarray:
    """
    Convert image to binary mask.

    Args:
        image (np.ndarray): Input image (BGR)

    Returns:
        np.ndarray: Binary mask
    """
    if image is None:
        raise ValueError("Input image is None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return mask