import cv2
import time
import torch

from ultralytics import YOLO

import paho.mqtt.client as mqtt

from Pipeline import detect_objects, extract_detected_objects, predict_text
from LicensePlateManager import process_plate_request, check_authorized_plates


# Gerät für Berechnungen festlegen
DEVICE = "cpu"

ACTIVE = False


# Funktion zur Verbindung mit dem MQTT-Broker
def connect_to_broker(client, broker_address, port):
    while True:
        try:
            client.connect(broker_address, port)
            print("Verbunden mit MQTT-Broker!")
            break
        except Exception as e:
            print(f"Verbindung fehlgeschlagen: {e}. Neuer Versuch in 5 Sekunden...")
            time.sleep(5)


# Funktion zum Abonieren von Topics
def subscribe_to_topics(client, topics):
    for topic in topics:
        client.subscribe(topic, qos=1)


# Callback-Funktion für erfolgreiche Anmeldung am Topic
def on_subscribe(client, userdata, mid, reason_code_list, properties):
    print(f"Erfolgreich abonniert")


# Callback-Funktion für erfolgreiche Veröffentlichung
def on_publish(client, userdata, mid, reason_code, properties):
    print(f"Nachricht erfolgreich veröffentlicht!")


# Callback-Funktion für empfangene Nachrichten
def on_message(client, userdata, message):
    print(
        f"Nachricht empfangen - Topic: {message.topic}, QoS: {message.qos}, Payload: {message.payload.decode('utf-8')}"
    )
    if message.topic == "PlateManager/Request":
        answer_plate_request(client, message.payload.decode("utf-8"))
    elif message.topic == "Location/Request":
        answer_location_request(client, message.payload.decode("utf-8"))


def answer_plate_request(client, message):
    """
    Diese Funktion bestimmt die Antwort der Request und sendet diese an
    den MQTT-Broker unter demselben Topic.
    """
    answer = process_plate_request(message)
    client.publish("PlateManager/Answer", str(answer), qos=1)


# Funktion zur Verarbeitung des Standorts
def answer_location_request(client, payload):
    global ACTIVE
    if payload == "True":
        ACTIVE = True
        client.publish("Location/Answer", "Kennzeichenerkennung aktiviert!", qos=1)
    else:
        ACTIVE = False
        client.publish("Location/Answer", "Kennzeichenerkennung deaktiviert!", qos=1)


# Funktion zur Veröffentlichung des Watchdogs
def publish_watchdog(client, topic):
    client.publish(topic, "Raspberry Pi läuft korrekt!", qos=1)


def publish_plate_recognition(client, topic, detection_model, recognition_model):
    # Hier Code zum Bild erstellen einfügen
    # Bild laden
    image = cv2.imread("test_images/image_08.png")

    # Objekte erkennen
    detected_boxes = detect_objects(detection_model, image)
    extracted_objects = extract_detected_objects(image, detected_boxes[0])

    # Erkannte Objekte speichern
    for i, obj in enumerate(extracted_objects):
        cv2.imwrite(f"output/detected_object_{i}.png", obj)

    # Texterkennung durchführen
    predicted_texts = [predict_text(recognition_model, obj) for obj in extracted_objects]

    print(predicted_texts)

    if check_authorized_plates(predicted_texts):
        client.publish(topic, "Authorisiertes Auto erkannt!", qos=1)
        time.sleep(30)


def main():
    global ACTIVE

    broker_address = "192.168.178.116"
    port = 1883

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_subscribe = on_subscribe
    client.on_publish = on_publish
    client.on_message = on_message

    connect_to_broker(client, broker_address, port)

    topics = ["PlateManager/Request", "Location/Request"]
    subscribe_to_topics(client, topics)

    # Modelle laden
    detection_model = YOLO("models/detection_model.pt")
    detection_model.eval()
    recognition_model = torch.load("models/recognition_model.pt", weights_only=False).to(DEVICE)
    recognition_model.eval()

    client.loop_start()

    i = 0
    while True:
        if ACTIVE:
            publish_plate_recognition(client, "Detection", detection_model, recognition_model)
            i += 10
        else:
            i += 1
            time.sleep(1)

        if i == 100:
            i = 0
            publish_watchdog(client, "Watchdog")


if __name__ == "__main__":
    main()
