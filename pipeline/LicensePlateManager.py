import json


class LicensePlateManager:
    def __init__(self, filename="authorized_plates.json"):
        """
        Initialisiert das Objekt mit dem angegebenen Dateinamen und lädt die
        Daten für autorisierte Fahrzeuge.
        """
        self.filename = filename
        self.authorized_plates = self.load_authorized_plates()

    def load_authorized_plates(self):
        """
        Lädt die autorisierten Fahrzeuge aus der JSON-Datei.
        Falls die Datei nicht existiert oder leer ist, wird ein leeres Dictionary zurückgegeben.
        """
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_authorized_plates(self):
        """
        Speichert die autorisierten Fahrzeuge in der JSON-Datei.
        """
        with open(self.filename, "w") as f:
            json.dump(self.authorized_plates, f, indent=4)

    def add_authorized_plate(self, plate):
        """
        Fügt ein Kennzeichen zur Liste der autorisierten Fahrzeuge hinzu.
        """
        if plate not in self.authorized_plates:
            self.authorized_plates.append(plate)
            self.save_authorized_plates()  # Speichert die geänderte Liste
            return f"Kennzeichen {plate} hinzugefügt."
        else:
            return f"Kennzeichen {plate} ist bereits autorisiert."

    def remove_authorized_plate(self, plate):
        """
        Entfernt ein Kennzeichen aus der Liste der autorisierten Fahrzeuge.
        """
        if plate in self.authorized_plates:
            self.authorized_plates.remove(plate)  # Entfernt das Kennzeichen
            self.save_authorized_plates()  # Speichert die geänderte Liste
            return f"Kennzeichen {plate} entfernt."
        else:
            return f"Kennzeichen {plate} nicht gefunden."

    def is_plate_authorized(self, plate):
        """
        Überprüft, ob ein Fahrzeug mit dem angegebenen Kennzeichen autorisiert ist.
        """
        return plate in self.authorized_plates

    def get_all_authorized_plates(self):
        """
        Gibt alle autorisierten Kennzeichen zurück.
        """
        return self.authorized_plates

    def clear_all_authorized_plates(self):
        """
        Entfernt alle Kennzeichen aus der Liste der autorisierten Fahrzeuge.
        """
        self.authorized_plates.clear()
        self.save_authorized_plates()
        return "Alle Kennzeichen wurden entfernt."


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


def check_authorized_plates(plates):
    authorized_plates_list = LicensePlateManager()

    for plate in plates:
        if authorized_plates_list.is_plate_authorized(plate):
            return True

    return False
