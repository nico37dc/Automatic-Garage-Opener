from PIL import Image
import os

# Pfad zum Ordner mit den Bildern
input_folder = "C:/Users/nicod/Documents/IoT/data/plates_bunt"
output_folder = "C:/Users/nicod/Documents/IoT/data/plates"

# Sicherstellen, dass der Ausgabeordner existiert
os.makedirs(output_folder, exist_ok=True)

# Zielgröße und Farbmodus
gray_mode = 'L'  # Graustufen

# Durchlaufe alle Dateien im Eingabeordner
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Konvertiere in Graustufen
        img_gray = img.convert(gray_mode)

        # Speichern im Ausgabeordner
        output_path = os.path.join(output_folder, filename)
        img_gray.save(output_path)

        print(f'Bearbeitet: {filename}')

print('Alle Bilder wurden verarbeitet.')