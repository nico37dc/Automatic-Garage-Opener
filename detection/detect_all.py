import cv2
import numpy as np
import os
from ultralytics import YOLO

def predict(model, image):
    results = model(image, conf=0.1)
    detected_objects = []
    for result in results:
        predicted_image = result.plot()
        boxes = result.boxes.xyxy.cpu().numpy()
        detected_objects.append(boxes)
    return predicted_image, detected_objects

def resize_and_pad_image(image, target_size=(128, 32), pad_color=100):
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    padded_image = np.full((target_h, target_w), pad_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = gray_image
    return padded_image

def extract_detections(image, boxes, output_folder, image_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_obj = image[y1:y2, x1:x2]
        processed_obj = resize_and_pad_image(cropped_obj)
        output_path = os.path.join(output_folder, f"{image_name}_object_{i}.png")
        cv2.imwrite(output_path, processed_obj)

def main():
    model = YOLO("best.pt")
    input_folder = "data/total/images"
    output_folder = "output_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        _, detected_objects = predict(model, image)

        if detected_objects:
            extract_detections(image, detected_objects[0], output_folder, image_name)

if __name__ == "__main__":
    main()
