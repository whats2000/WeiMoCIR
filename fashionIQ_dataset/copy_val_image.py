import os
import json
import shutil
from tqdm import tqdm

# Define paths
image_splits_folder = 'image_splits/'
images_folder = 'images/'
copy_image_folder = 'copy_image/'

# Ensure copy_image folder exists
os.makedirs(copy_image_folder, exist_ok=True)

# Collect all JSON files ending with 'val.json'
json_files = [f for f in os.listdir(image_splits_folder) if f.endswith('val.json')]

# Initialize a list to store all image names
all_images = []

# Process each JSON file with a progress bar
print("Reading JSON files...")
for json_file in tqdm(json_files, desc="Processing JSON files"):
    json_path = os.path.join(image_splits_folder, json_file)

    # Open and read JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
        all_images.extend(data)

# Remove duplicates by converting to set then back to list
all_images = list(set(all_images))

# Copy each image to the copy_image folder with progress bar
print("Copying images...")
for image_name in tqdm(all_images, desc="Copying Images", unit="image"):
    image_path = os.path.join(images_folder, image_name + '.png')
    if os.path.exists(image_path):
        shutil.copy(image_path, copy_image_folder)
    else:
        print(f'Image not found: {image_name}.jpg')

print(f"Total images copied: {len(all_images)}")
