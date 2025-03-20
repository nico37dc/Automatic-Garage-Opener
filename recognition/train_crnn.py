import os
import cv2
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from crnn import CRNN, custom_collate_fn
from augmentor import LicensePlateImageAugmentor
from dataset import LicensePlateDataset, LabelCodec


EPOCHS = 1000
BATCH_SIZE = 16
HIDDEN_LAYER = 64

MAX_TEXT_LEN = 10
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
NUM_CLASSES = len("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- ")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- DATENGENERATOR -------------------

# Verzeichnis mit deinen Bildern
PLATE_FOLDER = "data/custom_plates"
BACKGROUND_FOLDER = "data/backgrounds"

# Listen für Pfade und Labels
plate_images = []
plate_labels = []
background_images = []

# Alle Bilder im Ordner durchgehen
for filename in os.listdir(PLATE_FOLDER):
    if filename.endswith(".png"):
        plate_images.append(os.path.join(PLATE_FOLDER, filename))
        label = os.path.splitext(filename)[0]
        plate_labels.append(label)

for filename in sorted(os.listdir(BACKGROUND_FOLDER)):
    if filename.endswith(".png"):
        img_path = os.path.join(BACKGROUND_FOLDER, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        background_images.append(img)

background_images = np.array(background_images)

train_images, temp_images, train_labels, temp_labels = train_test_split(
    plate_images, plate_labels, test_size=0.3, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

augmentor = LicensePlateImageAugmentor(IMAGE_WIDTH, IMAGE_HEIGHT, background_images)

train_dataset = LicensePlateDataset(train_images, train_labels, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_TEXT_LEN, augmentor)
val_dataset = LicensePlateDataset(val_images, val_labels, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_TEXT_LEN, augmentor)
test_dataset = LicensePlateDataset(test_images, test_labels, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_TEXT_LEN, augmentor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)


# ------------------- MODELL -------------------
# BEIM NÄCHTEN TRAINING --> VORHANDENES MODELL LADEN
# model = CRNN(IMAGE_HEIGHT, 1, NUM_CLASSES, HIDDEN_LAYER).to(DEVICE)
model = torch.load("best_fine_custom.pt", weights_only=False).to(DEVICE)


# ------------------- LOSS & OPTIMIZER -------------------
# LERNRATE GGF. VERRRINGERN
criterion = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0002)


# ------------------- TRAINING -------------------
if __name__ == "__main__":
    best_accuracy = 0.00  # Variable für die beste Genauigkeit
    best_model_state = None  # Variable für den besten Modellzustand

    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch+1} started")
        train_loss = 0.00

        for images, labels, label_lengths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            outputs = outputs.log_softmax(2)

            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(DEVICE)

            loss = criterion(outputs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}")


        # ------------------- TEST ---------------------
        model.eval()
        correct_predictions = 0  # Zähler für korrekte Vorhersagen
        total_samples = 0  # Gesamtanzahl der getesteten Beispiele

        with torch.no_grad():
            for images, labels, label_lengths in test_loader:
                images = images.to(DEVICE)

                # Vorwärtsdurchlauf durch das Modell
                outputs = model(images)
                outputs = outputs.log_softmax(2)  # log softmax für probabilistische Werte

                # Vorhersage der maximalen Wahrscheinlichkeit pro Zeichen
                predictions = torch.argmax(outputs, dim=2)

                start_idx = 0  # Startindex für das erste Label
                for i in range(predictions.size(1)):  # Über alle Bilder in der Batch iterieren
                    # Extrahiere die Vorhersage für das aktuelle Bild
                    pred_indices = predictions[:, i].cpu().numpy()  # Nur relevante Vorhersagen
                    pred_text = LabelCodec.decode_prediction(pred_indices)  # Vorhersage dekodieren

                    # Extrahiere das wahre Label (Labels korrekt anhand der Längen aufteilen)
                    true_label = labels[start_idx : start_idx + label_lengths[i]].cpu().numpy()
                    true_text = LabelCodec.decode_number(true_label)  # Wahres Label dekodieren

                    # Startindex für das nächste Label anpassen
                    start_idx += label_lengths[i]

                    # Vergleich der Vorhersage mit dem echten Label
                    if pred_text == true_text:
                        correct_predictions += 1

                    total_samples += 1

                    # Ausgabe
                    if total_samples % 1 == 0:
                        print(f"Prediction: {pred_text} -- True: {true_text}")

        # Gesamtergebnis ausgeben
        accuracy = correct_predictions / total_samples
        print(f"Korrekt vorhergesagt: {correct_predictions} von {total_samples}")
        print(f"Genauigkeit: {accuracy:.2%}")

        # Modell speichern, wenn es das beste Modell ist
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # Speichern des besten Modells während des Trainings
            torch.save(model, "best_fine_custom_v2.pt")
            print(f"Bestes Modell gespeichert nach Epoch {epoch+1}!")

        # Das letzte Modell speichern
        torch.save(model, "last_fine_custom_v2.pt")
