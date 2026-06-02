import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input resized dataset
INPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\resized"

# Output augmented dataset
OUTPUT_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\augmented"

# Classes
classes = ["normal", "jaundice"]

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True
)

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

# Generate augmented images
for cls in classes:

    input_class_path = os.path.join(INPUT_PATH, cls)
    output_class_path = os.path.join(OUTPUT_PATH, cls)

    for image_name in os.listdir(input_class_path):

        image_path = os.path.join(input_class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))
        image = image.reshape((1,) + image.shape)

        # Save original image
        original_save_path = os.path.join(output_class_path, image_name)
        cv2.imwrite(original_save_path, image[0])

        # Generate augmented images
        aug_iter = datagen.flow(
            image,
            batch_size=1
        )

        for i in range(3):

            aug_image = next(aug_iter)[0].astype("uint8")

            save_path = os.path.join(
                output_class_path,
                f"{os.path.splitext(image_name)[0]}_aug{i}.jpg"
            )

            cv2.imwrite(save_path, aug_image)

print(" Data augmentation completed successfully.")