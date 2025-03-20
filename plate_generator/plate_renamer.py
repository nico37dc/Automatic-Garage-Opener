import os

# Entfernt "nummernschild-" am Anfang des Dateinamens

# Pfad zum Ordner mit den PNG-Dateien
folder_path = "C:/Users/nicod/Documents/IoT/data/plates"

# Durchlaufe alle Dateien im Ordner
for filename in os.listdir(folder_path):
    if filename.endswith(".png") and filename.startswith("nummernschild-"):
        new_filename = filename.replace("nummernschild-", "", 1)
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Umbenannt: {filename} -> {new_filename}")

print("Alle relevanten Dateien wurden umbenannt.")