import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8, random_state=42, subclasses=None, binary_classification=False, positive_classes=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): 'train' or 'test' to specify which split to use.
            train_ratio (float): Ratio of the dataset to be used as the training set.
            random_state (int): Seed for reproducibility of the train-test split.
            subclasses (list of str, optional): Specific subclasses to include (e.g., ['AZAN', 'HE', 'PAS']).
            binary_classification (bool): Whether to perform binary classification.
            positive_classes (list of str, optional): List of classes to be considered as positive in binary classification.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        self.binary_classification = binary_classification
        self.positive_classes = positive_classes if positive_classes else []

        # Gather all valid images and labels
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for sub_class in sorted(os.listdir(class_dir)):
                if subclasses is None or sub_class in subclasses:
                    sub_class_dir = os.path.join(class_dir, sub_class)
                    for img_name in sorted(os.listdir(sub_class_dir)):
                        img_path = os.path.join(sub_class_dir, img_name)

                        # Skip hidden files and handle file errors
                        if img_name.startswith("."):
                            continue
                        try:
                            # Test if the file is a valid image
                            with Image.open(img_path) as img:
                                img.verify()
                            self.image_paths.append(img_path)
                            # Assign binary or multi-class labels
                            if self.binary_classification:
                                self.labels.append(1 if class_name in self.positive_classes else 0)
                            else:
                                self.labels.append(idx)
                        except (UnidentifiedImageError, IOError):
                            print(f"Skipping invalid or unreadable file: {img_path}")

        # Stratified split based on labels
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.image_paths, self.labels, 
            stratify=self.labels, 
            train_size=train_ratio, 
            random_state=random_state
        )

        if split == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif split == 'test':
            self.image_paths = test_paths
            self.labels = test_labels
        else:
            raise ValueError("split should be 'train' or 'test'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Ensure image is loaded in RGB format and transform applied
        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, IOError):
            print(f"Error loading image {img_path}, skipping.")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label
