from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os


class SymlinkedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.root_dir = data_dir
        self.transform = transform

        # Get all class folders
        self.classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all file paths and labels
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for entry in os.scandir(cls_dir):
                if entry.is_file():
                    real_path = os.path.realpath(entry.path)  # Resolve symlinks
                    samples.append((real_path, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load the image
        image = Image.open(path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label
# class SymlinkedDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.root_dir = data_dir
#         self.transform = transform

#         # List all files, resolving symlinks to get the real paths
#         self.file_paths = [os.path.realpath(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         img_path = self.file_paths[idx]

#         # Load the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Apply any transformations
#         if self.transform:
#             image = self.transform(image)
        
#         return image, img_path


# class SymlinkedDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = Path(data_dir)
#         self.files = list(self.data_dir)  # or whatever extension
#         self.files = [f.resolve() for f in self.files]  # Resolve symlinks
#         self.transform = transform #or transforms.ToTensor()
        
#     def __len__(self):
#         return len(self.files)
        
#     def __getitem__(self, idx):
#         # Load the actual image
#         img_path = self.files[idx]
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
            
#         return image