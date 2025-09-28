import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from datasets.custom_dataset_csv_aug import CustomDatasetFromCSV
from torchvision import transforms
import torch
# --- CONFIG ---
csv_file = './dataset-mestrado-Gabriel/kfold_augmentations.csv'   # path to your CSV
data_dir = './dataset-mestrado-Gabriel'        # folder with images
fold = 0
split = "train"
image_size = 224

# Create dataset with default transform
dataset = CustomDatasetFromCSV(
    csv_file=csv_file,
    fold=fold,
    split=split,
    data_dir=data_dir,
    transform=CustomDatasetFromCSV.get_default_transform()
)

# Pick one sample
img, label = dataset[0]

# Convert back to numpy for display
def tensor_to_img(t):
    t = t.clone().detach()
    t = t.permute(1, 2, 0)  # CHW -> HWC
    t = t * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # unnormalize
    t = t.clamp(0, 1)
    return t.numpy()

# Show original + augmentations
sample_info = dataset.samples[0]
img_path = f"{data_dir}/{sample_info['image_path']}"

from PIL import Image
original = Image.open(img_path).convert("RGB")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(original)
axs[0].set_title("Original")
axs[0].axis("off")

# Apply each augmentation manually for visualization
for i, (name, aug) in enumerate(dataset.augmentation_transforms.items(), start=1):
    aug_img = aug(original)
    axs[i].imshow(aug_img)
    axs[i].set_title(name)
    axs[i].axis("off")

plt.show()
