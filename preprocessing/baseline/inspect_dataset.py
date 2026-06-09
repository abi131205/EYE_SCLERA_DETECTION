import os

# Dataset paths
DATASET_PATHS = {
    "Kaggle_Jaundice": r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\raw\kaggle_jaundice",
    "Zenodo_NJN": r"E:\Early_Jaundice_Risk_Prediction_DL\datasets\raw\zenodo_njn"
}

# Supported image formats
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

def inspect_dataset(dataset_name, dataset_path):
    print(f"\n📂 Inspecting Dataset: {dataset_name}")
    print("-" * 50)

    total_images = 0

    # Walk through folders
    for root, dirs, files in os.walk(dataset_path):

        image_files = [
            file for file in files
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS
        ]

        if image_files:
            folder_name = os.path.basename(root)

            print(f"\n Folder: {folder_name}")
            print(f" Number of Images: {len(image_files)}")

            total_images += len(image_files)

    print("\n Total Images Found:", total_images)


# Run inspection for all datasets
for dataset_name, dataset_path in DATASET_PATHS.items():
    inspect_dataset(dataset_name, dataset_path)

print("\n Dataset inspection completed successfully.")