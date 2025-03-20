import os

# Entfernt doppelte oder fehlerhafte Kennzeichen

# Pfad zum Ordner
folder_path = "C:/Users/nicod/Documents/IoT/data/plates"

# Durchlaufe alle Dateien im Ordner
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Überprüfe, ob es eine Datei ist (keine Unterordner berücksichtigen)
    if os.path.isfile(file_path):
        # Bedingung 1: Enthält eine "("
        contains_parenthesis = "(" in filename

        # Bedingung 2: Enthält maximal 1 "-"
        dash_count = filename.count("-")
        max_one_dash = dash_count <= 1

        # Wenn eine der Bedingungen erfüllt ist -> Löschen
        if contains_parenthesis or max_one_dash:
            os.remove(file_path)
            print(f"Gelöscht: {filename}")

print("Überprüfung abgeschlossen.")
