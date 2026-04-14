from smolagents import tool
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


@tool
def color_instances(mask_path: str, color_mode: str = "random") -> str:
    """
    Color each connected component (instance) in a binary mask.

    Args:
        mask_path (str): Path to the binary mask image.
        color_mode (str): Coloring scheme. Options:
            - "random": random colors for each instance
            - "pink": different shades of pink
            - "blue": different shades of blue

    Returns:
        str: Path to the saved colored image.
    """

    output_path = "outputs/colored.png"

    print("Loading mask from:", mask_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"❌ Failed to load mask: {mask_path}")

    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(mask)

    print("Number of detected instances:", num_labels)

    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    np.random.seed(42)

    for i in range(1, num_labels):  # skip background (label 0)

        if color_mode == "pink":
            color = get_pink_shade()

        elif color_mode == "blue":
            color = get_blue_shade()

        else:
            color = np.random.randint(0, 255, 3)

        colored[labels == i] = color

    success = cv2.imwrite(output_path, colored)

    if not success:
        raise IOError("❌ Failed to save colored image")

    print("✅ Saved colored image to:", output_path)

    return output_path