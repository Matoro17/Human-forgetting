import os
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, subclasses=None, allowed_classes=None, class_to_idx=None):
        """
        Args:
            root_dir (str): Directory with all the images, organized in class folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            subclasses (list of str, optional): Specific subclasses to include (e.g., ["AZAN", "HE", "PAS"]).
                                            If None, all subdirectories within a class are used.
            allowed_classes (list of str, optional): List of specific class names to include.
                                                If None, all directories in root_dir are considered classes.
            class_to_idx (dict, optional): A pre-defined mapping from class names to indices.
                                         If None, it will be created based on sorted allowed_classes or discovered classes.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Determine classes and class_to_idx mapping
        if allowed_classes:
            self.classes = sorted(allowed_classes)
            # Verify class directories exist
            for cls in self.classes:
                if not os.path.isdir(os.path.join(root_dir, cls)):
                    raise ValueError(f"Class directory {cls} not found in {root_dir}")
        else:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            # Verify provided mapping covers the determined classes
            if set(self.classes) != set(self.class_to_idx.keys()):
                raise ValueError("Provided class_to_idx does not match the classes found or specified.")

        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.num_classes = len(self.classes)

        # Gather all valid images and labels
        print(f"Loading dataset from: {root_dir}")
        print(f"Classes found/used: {self.classes}")
        print(f"Class mapping: {self.class_to_idx}")

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)

            # Determine subdirectories to scan
            subdirs_to_scan = []
            if subclasses:
                # Use only specified subclasses if they exist
                for sub_class in subclasses:
                    potential_subdir = os.path.join(class_dir, sub_class)
                    if os.path.isdir(potential_subdir):
                        subdirs_to_scan.append(potential_subdir)
                    else:
                        print(f"Warning: Specified subclass {sub_class} not found in {class_dir}")
            else:
                # Scan all subdirectories if no specific subclasses are given
                try:
                    items_in_class_dir = os.listdir(class_dir)
                except FileNotFoundError:
                    print(f"Warning: Class directory {class_dir} not found during scan.")
                    continue # Skip this class if directory doesn't exist

                for item in items_in_class_dir:
                    item_path = os.path.join(class_dir, item)
                    if os.path.isdir(item_path):
                        subdirs_to_scan.append(item_path)
                    # If no subdirs, maybe images are directly in class_dir?
                    elif os.path.isfile(item_path) and not subdirs_to_scan:
                         # Treat class_dir itself as the source if it contains files and no subdirs were found yet
                         # This check prevents adding class_dir multiple times
                         if class_dir not in subdirs_to_scan:
                             subdirs_to_scan.append(class_dir)
                             break # Stop adding once class_dir is added

            # If still no subdirs found (e.g., empty class dir, or only files but subclasses specified)
            if not subdirs_to_scan and not any(os.path.isfile(os.path.join(class_dir, item)) for item in os.listdir(class_dir)):
                 print(f"Warning: No subdirectories or image files found for class {class_name} with current settings.")
                 continue # Skip to next class
            elif not subdirs_to_scan and any(os.path.isfile(os.path.join(class_dir, item)) for item in os.listdir(class_dir)):
                 # Handle case where images are directly in the class folder and no subclasses were specified
                 if not subclasses:
                     subdirs_to_scan.append(class_dir)

            # Process images within the identified subdirectories (or the class directory itself)
            for sub_dir in subdirs_to_scan:
                try:
                    img_names = os.listdir(sub_dir)
                except FileNotFoundError:
                     print(f"Warning: Subdirectory {sub_dir} not found during scan.")
                     continue # Skip this subdir if it doesn't exist

                for img_name in sorted(img_names):
                    if img_name.startswith("."): # Skip hidden files
                        continue
                    img_path = os.path.join(sub_dir, img_name)
                    if os.path.isfile(img_path):
                        try:
                            # Test if the file is a valid image
                            with Image.open(img_path) as img:
                                img.verify() # Check if it's a valid image file
                            self.image_paths.append(img_path)
                            self.labels.append(class_idx)
                        except (UnidentifiedImageError, IOError, FileNotFoundError) as e:
                            print(f"Skipping invalid, unreadable, or missing file: {img_path} ({e})")
                        except Exception as e:
                            print(f"Skipping file due to unexpected error: {img_path} ({e})")

        print(f"Dataset loaded. Found {len(self.image_paths)} images.")
        # Print label distribution
        unique_labels, counts = torch.unique(torch.tensor(self.labels), return_counts=True)
        print("Label distribution:")
        for label_idx, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"  Class {label_idx} ({self.idx_to_class.get(label_idx, 'Unknown')}): {count} samples")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            print(f"Error loading image {img_path} at index {idx}: {e}. Returning None.")
            # Handle appropriately in the DataLoader or training loop, e.g., by skipping
            # For now, returning None might cause issues downstream if not handled.
            # A better approach might be to return a placeholder or skip this index in the DataLoader collate_fn.
            # Let's return the label anyway, maybe it's useful for skipping.
            return None, torch.tensor(label)
        except Exception as e:
            print(f"Unexpected error loading image {img_path} at index {idx}: {e}. Returning None.")
            return None, torch.tensor(label)

        if self.transform:
            # Apply the transform. The transform should handle generating multiple views if needed (e.g., for SSL).
            # For standard classification, it will just return one transformed image.
            transformed_image = self.transform(image)
        else:
            # Default transform if none provided
            transformed_image = transforms.ToTensor()(image)

        return transformed_image, torch.tensor(label)

