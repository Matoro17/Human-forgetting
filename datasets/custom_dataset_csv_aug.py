# datasets/custom_dataset_csv.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_file, fold, split, data_dir, transform=None, classes_to_exclude=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            fold (int): The fold number to use.
            split (string): 'train' or 'test'.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            classes_to_exclude (list, optional): List of classes to ignore.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Read the full CSV
        df = pd.read_csv(csv_file)
        
        # Filter by fold and split
        df = df[(df['fold'] == fold) & (df['split'] == split)]
        
        # Exclude specified classes
        if classes_to_exclude:
            df = df[~df['class_name'].isin(classes_to_exclude)]

        self.samples = df.to_dict('records')
        
        # Create a mapping from class name to integer label
        self.class_names = sorted(df['class_name'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Pre-define augmentation transforms
        self.augmentation_transforms = {
            'jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            'blur': transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            'solarize': transforms.RandomSolarize(threshold=128, p=1.0),
        }

    @staticmethod
    def get_default_transform(image_size=224):
        """Standard transform for validation, testing, and fine-tuning original images."""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        img_path = os.path.join(self.data_dir, sample_info['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        label_name = sample_info['class_name']
        label = self.class_to_idx[label_name]
        
        aug_type = sample_info.get('augmentation_type', 'none')

        # The main DINO transform (MultiCrop) is applied directly, ignoring CSV augmentations
        # This is because DINO pre-training has its own complex augmentation scheme.
        if hasattr(self.transform, 'global1'): # Heuristic to check if it's MultiCropTransform
             return self.transform(image), label

        # For fine-tuning and baseline training:
        final_transform = []
        
        # 1. Apply the specific augmentation if requested
        if aug_type in self.augmentation_transforms:
            final_transform.append(self.augmentation_transforms[aug_type])
            
        # 2. Apply the default transform (Resize, ToTensor, Normalize)
        if self.transform:
             # This assumes self.transform is the default_transform
            final_transform.extend(self.transform.transforms) 

        composed_transform = transforms.Compose(final_transform)
        
        return composed_transform(image), label

    @property
    def labels(self):
        """Returns a list of all labels in the dataset, used for samplers."""
        return [self.class_to_idx[s['class_name']] for s in self.samples]