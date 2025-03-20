import paho.mqtt.client as mqtt
import time
from mqtt_connection import answer_plate_request


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


# Callback-Funktion für erfolgreiche Anmeldung am Topic
def on_subscribe(client, userdata, mid, reason_code_list, properties):
    print(f"Erfolgreich abonniert")


# Callback-Funktion für empfangene Nachrichten
def on_message(client, userdata, message):
    print(
        f"Nachricht empfangen - Topic: {message.topic}, QoS: {message.qos}, Payload: {message.payload.decode('utf-8')}"
    )
    if message.topic == "LicensePlateManager/Request":
        answer_plate_request(client, message.payload.decode('utf-8'))


def main():
    broker_address = "192.168.178.116"
    port = 1883

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    connect_to_broker(client, broker_address, port)

    client.subscribe("LicensePlateManager/Request")

    client.loop_forever()


if __name__ == "__main__":
    main()
