import torch
import numpy as np
from PIL import Image
from augmentor import LicensePlateImageAugmentor
from dataset import LabelCodec
import cv2
import os

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
BACKGROUND_FOLDER = "data/backgrounds"

# Hintergrundbilder laden
background_images = []
for filename in sorted(os.listdir(BACKGROUND_FOLDER)):
    if filename.endswith(".png"):
        img_path = os.path.join(BACKGROUND_FOLDER, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        background_images.append(img)

background_images = np.array(background_images)

# Gerät (CPU oder GPU) festlegen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell laden
model = torch.load("best_fine.pt", weights_only=False).to(DEVICE)
model.eval()  # Modell in den Evaluierungsmodus versetzen

# Bild laden
image_path = "test6.png"
image = Image.open(image_path).convert("L")
plate_img = np.array(image)

print(plate_img, "\n")

# Augmentierung und Bildvorbereitung
augmentor = LicensePlateImageAugmentor(IMAGE_WIDTH, IMAGE_HEIGHT, background_images)
augmented_image = augmentor.generate_plate_image(plate_img)

print(augmented_image, "\n")


# Normalisierung und Vorverarbeitung des Bildes
augmented_image = augmented_image.astype(np.float32) / 255.0  # Normalisierung auf [0, 1]

print(augmented_image, "\n")

augmented_image = np.expand_dims(augmented_image, axis=0)  # Batch-Dimension hinzufügen

print(augmented_image, "\n")

augmented_image = torch.from_numpy(augmented_image).unsqueeze(0).to(DEVICE)  # [1, 1, 32, 128]

# Bild durch das Modell laufen lassen
with torch.no_grad():  # Keine Gradientenberechnung während der Inferenz
    outputs = model(augmented_image)
    outputs = outputs.log_softmax(2)  # log softmax für probabilistische Werte

    # Vorhersage der maximalen Wahrscheinlichkeit pro Zeichen
    predictions = torch.argmax(outputs, dim=2)

    # Extrahiere die Vorhersage für das Bild
    pred_indices = predictions.squeeze(0).cpu().numpy()  # Entferne Batch-Dimension und hole Vorhersage
    pred_text = LabelCodec.decode_prediction(pred_indices)  # Vorhersage dekodieren

# Ausgabe der Vorhersage
print(f"Prediction for the single image: {pred_text}")
