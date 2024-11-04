import os
import sys
import csv

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from datasets.custom_dataset import CustomDataset


import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def count_images_per_class(dataset_root):
    # Initialize a dictionary to store class and subclass image counts
    class_counts = {}

    # Loop through each class (folder) in the dataset root directory
    for class_name in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.isdir(class_dir):
            subclass_counts = {}
            total_class_images = 0

            # Loop through each subclass (subfolder) in the class directory
            for subclass_name in sorted(os.listdir(class_dir)):
                subclass_dir = os.path.join(class_dir, subclass_name)
                if os.path.isdir(subclass_dir):
                    image_count = len([img for img in os.listdir(subclass_dir) if img.endswith(('jpg', 'jpeg', 'png'))])
                    subclass_counts[subclass_name] = image_count
                    total_class_images += image_count

            # Store the counts in the class_counts dictionary
            class_counts[class_name] = {
                'total_images': total_class_images,
                'subclasses': subclass_counts
            }

    return class_counts

def export_class_counts_to_csv(class_counts, output_file):
    # Write the data to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Class', 'Total Images', 'Subclass', 'Subclass Image Count'])

        # Write the data
        for class_name, class_data in class_counts.items():
            # Write class row with total images, leave subclass columns empty
            writer.writerow([class_name, class_data['total_images'], '', ''])
            # Write each subclass row, leave class and total images columns empty
            for subclass_name, count in class_data['subclasses'].items():
                writer.writerow(['', '', subclass_name, count])

if __name__ == "__main__":
    dataset_root = DATASET_DIR  # Specify the path to your dataset root
    output_csv = 'class_counts.csv'  # Specify the output CSV file name

    # Count images per class and subclass
    class_counts = count_images_per_class(dataset_root)

    # Export the counts to a CSV file
    export_class_counts_to_csv(class_counts, output_csv)

    print(f"Class counts have been exported to {output_csv}")
