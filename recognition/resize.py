import cv2
import numpy as np

def resize_and_pad_image(image, target_size=(128, 32), pad_color=100):
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Skaliert das Bild proportional
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Konvertiere das Bild in Graustufen
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Erstelle das gepaddete Bild mit der Zielgröße
    padded_image = np.full((target_h, target_w), pad_color, dtype=np.uint8)

    # Bestimme die Position für das Bild (mittig platzieren)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Kopiere das skalierte und kontrastierte Bild in das gepaddete Bild
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = gray_image

    return padded_image

image = cv2.imread("test5.png")

new_image = resize_and_pad_image(image)

cv2.imwrite("test5.png", new_image)