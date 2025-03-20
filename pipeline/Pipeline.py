import cv2
import torch
import numpy as np
from ultralytics import YOLO

from crnn import LabelCodec

# Gerät für Berechnungen festlegen
DEVICE = "cpu"


def detect_objects(model, image):
    """Führt eine Objekterkennung auf dem Bild durch."""
    results = model(image, conf=0.5)
    detected_boxes = []

    for result in results:
        # Debugging:
        predicted_image = result.plot()
        cv2.imwrite("output/predicted_image.png", predicted_image)

        # Bounding-Box-Koordinaten extrahieren
        boxes = result.boxes.xyxy.cpu().numpy()
        detected_boxes.append(boxes)

    return detected_boxes


def resize_and_pad(image, target_size=(128, 32), pad_color=100):
    """Skaliert das Bild proportional und füllt es auf die Zielgröße auf."""
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Bild proportional skalieren
    scale = min(target_w / w, target_h / h)
    if w / h > 2.5 and scale > 0.1:
        scale = scale - 0.1

    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # In Graustufen umwandeln
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Gepaddetes Bild erstellen
    padded_image = np.full((target_h, target_w), pad_color, dtype=np.uint8)

    # Position zum Einfügen bestimmen
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Skalierte Version in das gepaddete Bild kopieren
    padded_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = grayscale_image

    return padded_image


def extract_detected_objects(image, boxes):
    """Extrahiert erkannte Objekte basierend auf den Bounding-Box-Koordinaten."""
    extracted_objects = []

    for x1, y1, x2, y2 in boxes:
        # Koordinaten runden
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Objekt aus dem Bild ausschneiden
        cropped_object = image[y1:y2, x1:x2]

        # Skaliertes und gepaddetes Bild erstellen
        processed_object = resize_and_pad(cropped_object)
        extracted_objects.append(processed_object)

    return extracted_objects


def predict_text(model, image):
    """Führt eine Texterkennung auf dem Bild durch."""
    image = np.array(image).astype(np.float32) / (255.0**2)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image).log_softmax(2)
        predicted_indices = torch.argmax(output, dim=2).squeeze(0).cpu().numpy()
        predicted_text = LabelCodec.decode_prediction(predicted_indices)

    return predicted_text
