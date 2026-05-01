import numpy as np
import cv2


def expand(mask: np.ndarray, pixels: int) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask cannot be None")

    if pixels <= 0:
        return mask

    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)