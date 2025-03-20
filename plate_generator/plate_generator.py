import time
import string
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# Liest die Kennzeichen-Codes aus einer CSV-Datei und bereinigt die Daten
def get_plate_codes():
    counties_df = pd.read_csv("plates_germany.csv", delimiter=";")
    counties_df["Autokennzeichen"] = counties_df["Autokennzeichen"].str.replace("*", "")
    plate_codes = counties_df["Autokennzeichen"].dropna().tolist()
    # print(plate_codes)
    # print(len(plate_codes))
    return plate_codes


# Generiert eine zufällige Zeichenkette (Buchstaben) mit einer maximalen Länge
def generate_random_letters(max_length):
    length = random.randint(1, max_length)
    return "".join(random.choices(string.ascii_uppercase, k=length))


# Generiert eine zufällige Zeichenkette (Zahlen) mit einer maximalen Länge
def generate_random_digits(max_length, plate_type):
    length = random.randint(1, max_length)
    digits = "".join(random.choices(string.digits, k=length))
    if plate_type == "B":
        return digits
    else:
        return digits + plate_type


# Initialisiert den Webdriver und öffnet die Zielwebsite
def setup_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get("https://onlinestreet.de/kennzeichen/generator")

    ok_button = driver.find_element(By.CLASS_NAME, "fc-cta-consent")
    ok_button.click()
    return driver


# Generiert ein Kennzeichenbild basierend auf den Zufallswerten und lädt es herunter
def generate_and_download_plate(driver, plate_code, plate_type, iteration):
    remaining_length = 8 - len(plate_code)

    if plate_type == "B":
        max_letters_length = min(2, remaining_length - 1)
        max_number_length = min(4, remaining_length - max_letters_length)
    else:
        max_letters_length = min(2, remaining_length - 2)
        max_number_length = min(4, remaining_length - max_letters_length - 1)

    input_field1 = driver.find_element(By.NAME, "1")
    input_field2 = driver.find_element(By.NAME, "2")
    input_field3 = driver.find_element(By.NAME, "3")

    random_letters = generate_random_letters(max_letters_length)
    random_digits = generate_random_digits(max_number_length, plate_type)

    input_field1.clear()
    input_field1.send_keys(plate_code)

    input_field2.clear()
    input_field2.send_keys(random_letters)

    input_field3.clear()
    input_field3.send_keys(random_digits)

    generate_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Kennzeichen generieren')]")))
    generate_button.click()

    download_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[text()='200 Pixel']")))
    download_button.click()

    print(f"Iteration: {iteration}\t Plate: {plate_code}\t Letters: {random_letters}\t Digits: {random_digits}")
    time.sleep(0.2)


# Iteriert über die Kennzeichenliste und generiert mehrere Bilder pro Kennzeichen
def generate_plates(plate_codes, plate_number, plate_type):
    driver = setup_driver()
    try:
        for plate_code in plate_codes:
            for iteration in range(plate_number):
                generate_and_download_plate(driver, plate_code, plate_type, iteration)
    finally:
        time.sleep(1)
        driver.quit()


# Hauptfunktion zum Starten des Ablaufs
def main():
    # "B" -> Basic, "E" -> Electric, "H" -> Historic
    plate_type = "B"
    plate_number = 100
    plate_codes = get_plate_codes()
    generate_plates(plate_codes, plate_number, plate_type)


if __name__ == "__main__":
    main()
