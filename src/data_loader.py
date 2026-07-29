import os

import numpy as np
from PIL import Image


def load_traffic_data(data_dir, target_size=(224, 224)):
    """
    Load, resize, and convert traffic sign images 
    from disk into preprocessed arrays and labels.
    """

    images = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"⚠️ Error: The directory '{data_dir}' does not exist.")
        return np.array(images), np.array(labels)

    # Loop through each UK sign category folder
    for folder_name in os.listdir(data_dir):
        category_path = os.path.join(data_dir, folder_name)

        if os.path.isdir(category_path):
            uk_folder_path = os.path.join(category_path, "United_Kingdom")

            if os.path.isdir(uk_folder_path):
                print(f"💿 Loading UK images from category: {folder_name}")

                for img_name in os.listdir(uk_folder_path):
                    # Handles jpg/png files
                    if img_name.lower().endswith((
                        '.png', 
                        '.jpg', 
                        '.webp', 
                        '.jpeg'
                    )):
                        try:
                            img_path = os.path.join(uk_folder_path, img_name)
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(target_size)

                            images.append(np.array(img))
                            # Use folder name as the text label
                            labels.append(folder_name)
                        except (ValueError, RuntimeError) as e:
                            print(f"❌ Skipping broken image {img_name}: {e}")

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    DATA_PATH = os.path.join(BASE_DIR, "data", "dataset")

    print(f"Looking for data in: {DATA_PATH}")

    X, y = load_traffic_data(DATA_PATH)

    print(f"✅\nSuccessfully loaded {X.shape[0]} images!")
    print(f"Image Shape: {X.shape[1:]} (Width x Height x Channels)")