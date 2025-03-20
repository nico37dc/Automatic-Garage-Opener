import torch
import numpy as np
from PIL import Image
from dataset import LabelCodec


# Gerät (CPU oder GPU) festlegen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modell laden
model = torch.load("best_fine.pt", weights_only=False).to(DEVICE)
model.eval()  # Modell in den Evaluierungsmodus versetzen

# Bild laden
image_path = "test6.png"
image = Image.open(image_path).convert("L")
image = np.array(image)

print(image, "\n")

# Augmentor nachbilden
image = image.astype(np.float32) / 255.0

print(image, "\n")

image = image.astype(np.float32) / 255.0

print(image, "\n")

image = np.expand_dims(image, axis=0)

print(image, "\n")

augmented_image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

print(augmented_image, "\n")


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
