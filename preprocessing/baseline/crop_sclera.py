import os
import cv2

# Input dataset path
INPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\raw\kaggle_jaundice\archive (3)"

# Output cropped path
OUTPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\cropped"

# Haarcascade file
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# Load eye detector
eye_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Create output folders
classes = ["normal", "jaundice"]

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# Process images
for cls in classes:

    class_input_path = os.path.join(INPUT_PATH, cls)
    class_output_path = os.path.join(OUTPUT_PATH, cls)

    for image_name in os.listdir(class_input_path):

        image_path = os.path.join(class_input_path, image_name)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Crop first detected eye
        for i, (x, y, w, h) in enumerate(eyes):

            eye_crop = image[y:y+h, x:x+w]

            save_path = os.path.join(
                class_output_path,
                f"{os.path.splitext(image_name)[0]}_eye{i}.jpg"
            )

            cv2.imwrite(save_path, eye_crop)

            # Save only first eye
            break

print(" Eye cropping completed successfully.")