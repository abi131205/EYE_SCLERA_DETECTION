import os
import cv2

# Input cropped dataset
INPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\cropped"

# Output resized dataset
OUTPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\resized"

# Resize dimensions
IMAGE_SIZE = (128, 128)

# Classes
classes = ["normal", "jaundice"]

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# Resize process
for cls in classes:

    input_class_path = os.path.join(INPUT_PATH, cls)
    output_class_path = os.path.join(OUTPUT_PATH, cls)

    for image_name in os.listdir(input_class_path):

        image_path = os.path.join(input_class_path, image_name)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        resized_image = cv2.resize(image, IMAGE_SIZE)

        # Save resized image
        save_path = os.path.join(output_class_path, image_name)

        cv2.imwrite(save_path, resized_image)

print(" Image resizing completed successfully.")