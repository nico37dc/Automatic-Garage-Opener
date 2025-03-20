import torch
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LabelCodec:
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

    @staticmethod
    def encode_number(number):
        return list(map(lambda c: LabelCodec.ALPHABET.index(c), number))

    @staticmethod
    def decode_number(label):
        return "".join(list(map(lambda x: LabelCodec.ALPHABET[int(x)], label)))

    @staticmethod
    def decode_prediction(prediction):
        out_best = prediction.squeeze()

        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ""
        for c in out_best:
            if c < len(LabelCodec.ALPHABET) - 1:
                outstr += LabelCodec.ALPHABET[c]
        return outstr


class LicensePlateDataset(Dataset):
    def __init__(self, images, labels, img_w, img_h, max_text_len, augmentor):
        self.images = images
        self.labels = labels
        self.img_w = img_w
        self.img_h = img_h
        self.max_text_len = max_text_len
        self.augmentor = augmentor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Lade das Bild, hier wird angenommen, dass `images` Pfade sind
        image_path = self.images[idx]
        image_label = self.labels[idx]

        # Bild laden
        plate_img = Image.open(image_path).convert("L")  # Konvertiere zu Graustufen
        plate_img = np.array(plate_img)

        # Augmentierung und Bildvorbereitung
        # image = self.augmentor.generate_plate_image(plate_img)
        image = plate_img.astype(np.float32) / 255.0

        # Für Custom Datensatz

        # debug
        # if isinstance(image, np.ndarray):
        #     # Falls das Bild Float-Werte enthält, erst in uint8 umwandeln
        #     if image.dtype in [np.float32, np.float64]:
        #         debug_image = (image * 255).clip(0, 255).astype(np.uint8)
        #     else:
        #         debug_image = image.copy()

        #     debug_image = Image.fromarray(debug_image)

        # # Nur die Kopie speichern, das Original bleibt unverändert
        # debug_image.save("debug.png")

        image = np.expand_dims(image, -1)
        image = image.astype(np.float32) / 255.0  # Normalisierung

        # Bild zu Tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Label kodieren
        label_encoded = LabelCodec.encode_number(image_label)
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return image, label_tensor
