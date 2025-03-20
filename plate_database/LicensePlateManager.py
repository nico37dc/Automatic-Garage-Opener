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
        self.authorized_plates.clear()  # Löscht alle Einträge in der Liste
        self.save_authorized_plates()  # Speichert die leere Liste
        return "Alle Kennzeichen wurden entfernt."


def main():
    # Beispielverwendung:
    authorized_plates_list = LicensePlateManager()

    # Kennzeichen hinzufügen
    print(authorized_plates_list.add_authorized_plate("B-XY1234"))
    print(authorized_plates_list.add_authorized_plate("M-AB5678"))

    # Überprüfen, ob ein Fahrzeug autorisiert ist
    print(authorized_plates_list.is_plate_authorized("B-XY1234"))  # True
    print(authorized_plates_list.is_plate_authorized("M-XD0000"))  # False

    # Fahrzeug entfernen
    print(authorized_plates_list.remove_authorized_plate("B-XY1234"))

    # Alle autorisierten Kennzeichen anzeigen
    print(authorized_plates_list.get_all_authorized_plates())

    # Alle Kennzeichen entfernen
    print(authorized_plates_list.clear_all_authorized_plates())

    # Alle autorisierten Kennzeichen anzeigen
    print(authorized_plates_list.get_all_authorized_plates())


if __name__ == "__main__":
    main()
