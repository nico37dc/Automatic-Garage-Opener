from LicensePlateManager import LicensePlateManager


def answer_plate_request(client, message):
    """
    Diese Funktion bestimmt die Antwort der Request und sendet diese an
    den MQTT-Broker unter demselben Topic.
    """
    answer = process_plate_request(message)
    client.publish("LicensePlateManager/Answer", str(answer), qos=1)


def process_plate_request(message):
    """
    Diese Funktion verarbeitet einen Befehl und führt je nach Inhalt eine der
    vorgesehenen Aktionen aus: Hinzufügen, Entfernen, Anzeigen oder Löschen.
    """
    authorized_plates_list = LicensePlateManager()

    # Befehl in Teile zerlegen
    command_parts = message.split()

    if not command_parts:
        return "Ungültiger Befehl!"

    # Überprüfen, welcher Befehl ausgeführt werden soll
    if command_parts[0] == "add" and len(command_parts) == 2:
        plate = command_parts[1]
        # Funktion zum Hinzufügen eines Fahrzeugs aufrufen
        return authorized_plates_list.add_authorized_plate(plate)

    elif command_parts[0] == "remove" and len(command_parts) == 2:
        plate = command_parts[1]
        # Funktion zum Entfernen eines Fahrzeugs aufrufen
        return authorized_plates_list.remove_authorized_plate(plate)

    elif command_parts[0] == "show" and command_parts[1] == "all":
        # Alle Fahrzeuge anzeigen
        return authorized_plates_list.get_all_authorized_plates()

    elif command_parts[0] == "clear" and command_parts[1] == "all":
        # Alle Fahrzeuge löschen
        return authorized_plates_list.clear_all_authorized_plates()

    else:
        return "Ungültiger Befehl oder falsche Argumente!"


def main():
    print(process_plate_request("add UL-CK334"))
    print(process_plate_request("show all"))
    print(process_plate_request("clear all"))


if __name__ == "__main__":
    main()
