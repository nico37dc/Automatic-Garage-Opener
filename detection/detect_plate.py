import cv2
import numpy as np
from ultralytics import YOLO


def predict(model, image):
    # Run batched inference on a list of images
    results = model(image, conf=0.1)

    detected_objects = []
    for result in results:
        predicted_image = result.plot()  # Annotiertes Bild
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding Boxes
        detected_objects.append(boxes)

    return predicted_image, detected_objects


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


def extract_detections(image, boxes):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Bounding Box-Koordinaten in Integer umwandeln
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Objekt ausschneiden
        cropped_obj = image[y1:y2, x1:x2]

        # Skaliertes, gepaddetes und kontrastiertes Bild erstellen
        processed_obj = resize_and_pad_image(cropped_obj)

        # Speichern
        cv2.imwrite(f"detected_object_{i}.png", processed_obj)


def main():
    # Modell laden
    model = YOLO("best.pt")

    # Bild laden
    image = cv2.imread("image4.png")

    # Vorhersage
    predicted_image, detected_objects = predict(model, image)

    cv2.imwrite("predicted_image.png", predicted_image)

    # Objekte extrahieren und speichern
    extract_detections(image, detected_objects[0])


if __name__ == "__main__":
    main()
