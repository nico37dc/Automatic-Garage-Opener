from ultralytics import YOLO


def train_model():
    # YOLO-Modell laden
    model = YOLO("last.pt")

    # Modell trainieren
    model.train(
        data="C:/Users/nicod/Documents/IoT/detection/data/dataset/data.yaml",
        epochs=1000,
        patience=0,
        batch=32,
        lr0=0.0001,
        optimizer="AdamW",
    )


if __name__ == "__main__":
    train_model()
