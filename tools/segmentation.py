import cv2

def run_segmentation(image_path: str, output_path: str):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    overlay = img.copy()
    overlay[mask > 0] = [0, 0, 255]

    cv2.imwrite(output_path, overlay)

    return output_path