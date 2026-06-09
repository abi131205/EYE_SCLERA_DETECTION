import os
from PIL import Image

DATASET_PATH = r"datasets/cropped"

total_images = 0
corrupt_images = []

print("\nChecking Dataset...\n")

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    image_count = 0

    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        try:
            img = Image.open(file_path)
            img.verify()
            image_count += 1

        except Exception:
            corrupt_images.append(file_path)

    total_images += image_count

    print(f"{class_name}: {image_count} images")

print("\n---------------------")
print(f"Total Images: {total_images}")
print(f"Corrupt Images: {len(corrupt_images)}")

if corrupt_images:
    print("\nCorrupt Files:")
    for img in corrupt_images:
        print(img)