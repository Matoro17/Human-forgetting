import os
from PIL import Image

def get_image_sizes(dataset_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    sizes = []
    total_images = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                try:
                    image_path = os.path.join(root, file)
                    with Image.open(image_path) as img:
                        sizes.append(img.size)  # (width, height)
                        total_images += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not sizes:
        print("No images found.")
        return

    widths, heights = zip(*sizes)

    print(f"\n📊 Image Size Summary for: {dataset_path}")
    print(f"Total images processed: {total_images}")
    print(f"Minimum width: {min(widths)} px")
    print(f"Maximum width: {max(widths)} px")
    print(f"Minimum height: {min(heights)} px")
    print(f"Maximum height: {max(heights)} px")
    print(f"\n🧩 Unique image sizes (width x height):")

# Example usage
if __name__ == "__main__":
    dataset_folder = "dataset-mestrado-Gabriel"
    get_image_sizes(dataset_folder)
