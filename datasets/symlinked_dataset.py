from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os


class SymlinkedDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_order=None, 
                 binary_classification=False, positive_classes=None):
        """
        Args:
            data_dir (str): Root directory containing class folders
            transform (callable, optional): Transform to apply to images
            class_order (list, optional): Enforce specific class order
            binary_classification (bool): Whether to use binary labels
            positive_classes (list): Classes to treat as positive (for binary mode)
        """
        self.root_dir = data_dir
        self.transform = transform
        self.binary_classification = binary_classification
        self.positive_classes = positive_classes or []

        # Get all class folders
        detected_classes = [entry.name for entry in os.scandir(data_dir) if entry.is_dir()]
        
        # Validate and set class order
        if class_order:
            assert set(class_order) == set(detected_classes), \
                   f"Class mismatch. Expected {class_order}, found {detected_classes}"
            self.classes = class_order
        else:
            self.classes = sorted(detected_classes)

        # Validate binary classification settings
        if self.binary_classification:
            if not self.positive_classes:
                raise ValueError("positive_classes must be specified for binary classification")
            invalid = [pc for pc in self.positive_classes if pc not in self.classes]
            if invalid:
                raise ValueError(f"positive_classes {invalid} not found in dataset classes")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for entry in os.scandir(cls_dir):
                if entry.is_file():
                    real_path = os.path.realpath(entry.path)  # Resolve symlinks
                    
                    # Determine label
                    if self.binary_classification:
                        label = 1 if cls_name in self.positive_classes else 0
                    else:
                        label = self.class_to_idx[cls_name]
                    
                    samples.append((real_path, label))
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