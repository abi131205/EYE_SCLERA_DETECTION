import os
from PIL import Image

# Raw dataset path
RAW_DATASET_PATH = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\raw"

# Supported image formats
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Store corrupted files
corrupted_files = []

def check_images(dataset_path):

    print(f"\n Checking Dataset: {dataset_path}")

    total_images = 0
    valid_images = 0

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            file_extension = os.path.splitext(file)[1].lower()

            if file_extension in IMAGE_EXTENSIONS:

                total_images += 1

                file_path = os.path.join(root, file)

                try:
                    # Open image to verify
                    img = Image.open(file_path)
                    img.verify()

                    valid_images += 1

                except Exception as e:
                    corrupted_files.append(file_path)

    print(f" Total Images: {total_images}")
    print(f" Valid Images: {valid_images}")
    print(f" Corrupted Images: {len(corrupted_files)}")


# Run checking
check_images(RAW_DATASET_PATH)

# Print corrupted files
if corrupted_files:

    print("\n Corrupted Files Found:")

    for file in corrupted_files:
        print(file)

else:
    print("\n No corrupted images found.")