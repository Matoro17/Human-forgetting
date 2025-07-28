import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDatasetFromCSV(Dataset):
    def __init__(self,
                 transform=None,
                 csv_file=None,
                 fold=None,
                 split='train',
                 binary_classification=False,
                 positive_classes=None,
                 classes_to_exclude=None,
                 data_dir='.'): # New parameter for classes to exclude
        """
        Args:
            transform (callable): Transformations to apply to images.
            csv_file (str): Path to the CSV file with columns: fold, split, class_name, image_path.
            fold (int): Fold number to select data from.
            split (str): 'train' or 'test'.
            binary_classification (bool): Use binary labels (0/1).
            positive_classes (list): Class names to be treated as positive.
            classes_to_exclude (list): List of class names to exclude from the dataset.
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.binary_classification = binary_classification
        self.positive_classes = positive_classes or []
        self.classes_to_exclude = classes_to_exclude or []
        self.data_dir = data_dir

        if csv_file is not None:
            self._load_from_csv(csv_file, fold, split)
        else:
            raise ValueError("csv_file must be provided for CSV-based loading.")

    def _load_from_csv(self, csv_file, fold, split):
        class_names_in_fold = set()
        all_rows_for_fold_split = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by fold and split first
                if int(row['fold']) == fold and row['split'] == split:
                    # Then filter by classes to exclude
                    if row['class_name'] not in self.classes_to_exclude:
                        all_rows_for_fold_split.append(row)
                        class_names_in_fold.add(row['class_name'])

        self.classes = sorted(list(class_names_in_fold))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for row in all_rows_for_fold_split:
            img_path = row['image_path']
            class_name = row['class_name']
            
            # Ensure the class_name exists in the current fold's classes before mapping
            if class_name in self.class_to_idx:
                label = 1 if (self.binary_classification and class_name in self.positive_classes) else self.class_to_idx[class_name]
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        relative_path = self.image_paths[idx]
        # The line below is the crucial change
        path = os.path.join(self.data_dir, relative_path) # <-- USE os.path.join
        
        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError:
            print(f"ERROR: Image not found at path: {path}")
            raise
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            views = self.transform(image)
            if isinstance(views, list):  # MultiCropTransform returns multiple views
                return views, torch.tensor(label)
            else:
                # For single-view transformations, return a stacked tensor to match DINO's expectation
                # If DINO expects a list even for single-crop, this might need adjustment in DINO trainer
                # but typically a single batch of images is sufficient.
                # Assuming DINOTrainer expects a batch of images for fine-tuning/evaluation,
                # a single transformed image is fine here.
                return views, torch.tensor(label)
        else:
            tensor = transforms.ToTensor()(image)
            # If no transform, return a single tensor, not stacked
            return tensor, torch.tensor(label) 

    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])