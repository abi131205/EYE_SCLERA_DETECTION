import os
import shutil
import random

# Input augmented dataset
SOURCE_DIR = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\augmented"

# Output split dataset
OUTPUT_DIR = r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\split"

# Classes
classes = ["normal", "jaundice"]

# Split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:

    source_path = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(source_path)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }.items():

        split_folder = os.path.join(OUTPUT_DIR, split_name, cls)

        os.makedirs(split_folder, exist_ok=True)

        for image in split_images:

            src = os.path.join(source_path, image)
            dst = os.path.join(split_folder, image)

            shutil.copy(src, dst)

    print(f" {cls} dataset split completed.")

print("\n Dataset splitting completed successfully.")