# create_kfold_csv.py
import os
import csv
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from PIL import Image, UnidentifiedImageError
from collections import Counter
import random

# --- List of augmentations to be applied for oversampling ---
AUGMENTATION_TYPES = ['jitter', 'blur', 'solarize']

def collect_nested_samples(data_dir, ignore_classes=None):
    """This function remains the same as your original."""
    samples = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    if ignore_classes:
        classes = [c for c in classes if c not in ignore_classes]

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for subclass in os.listdir(cls_dir):
            subclass_dir = os.path.join(cls_dir, subclass)
            if not os.path.isdir(subclass_dir):
                continue
            for fname in os.listdir(subclass_dir):
                if fname.startswith("."):
                    continue
                img_path = os.path.join(subclass_dir, fname)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    unique_path = os.path.relpath(img_path, data_dir)
                    samples.append((unique_path, cls, class_to_idx[cls]))
                except (UnidentifiedImageError, IOError):
                    print(f"Skipping unreadable image: {img_path}")
    return samples

def oversample_train_set(train_indices, labels):
    """
    MODIFIED: Return new training samples with specified augmentations.
    Instead of duplicating indices, it returns tuples of (index, augmentation_type).
    """
    # Group indices by class
    class_to_indices = {}
    for idx in train_indices:
        label = labels[idx]
        class_to_indices.setdefault(label, []).append(idx)

    # Find max class size
    if not class_to_indices:
        return [], {}
    max_size = max(len(idxs) for idxs in class_to_indices.values())

    # Oversample
    new_train_samples = []
    replication_info = {}
    
    for label, idxs in class_to_indices.items():
        # Add all original images first
        for idx in idxs:
            new_train_samples.append((idx, 'original'))
        
        current_size = len(idxs)
        samples_to_add = max_size - current_size
        
        if samples_to_add > 0:
            # Generate augmented samples by cycling through available augmentations
            replicated_indices = []
            for i in range(samples_to_add):
                original_idx = idxs[i % len(idxs)] # Pick an original image to augment
                aug_type = AUGMENTATION_TYPES[i % len(AUGMENTATION_TYPES)] # Cycle through augmentations
                replicated_indices.append((original_idx, aug_type))
            
            new_train_samples.extend(replicated_indices)
        
        replication_info[label] = {
            "original_count": current_size,
            "augmented_count": samples_to_add,
            "new_total": current_size + samples_to_add
        }

    random.shuffle(new_train_samples)
    return new_train_samples, replication_info

def create_kfold_csv(samples, k, output_csv, oversample=False):
    """
    MODIFIED: Writes the new `augmentation_type` column to the CSV.
    """
    image_paths = [s[0] for s in samples]
    class_names = [s[1] for s in samples]
    labels = [s[2] for s in samples]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    rows = []

    print(f"Total samples: {len(samples)}")
    print(f"Overall class distribution: {Counter(class_names)}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n--- Fold {fold}/{k-1} ---")
        
        train_samples_to_write = []
        if oversample:
            train_samples_augmented, replication_info = oversample_train_set(train_idx, labels)
            train_samples_to_write = train_samples_augmented
            
            oversampled_distribution = Counter(class_names[i] for i, aug_type in train_samples_to_write)
            print(f"Train samples AFTER oversampling: {len(train_samples_to_write)}")
            print(f"Train class distribution (AFTER oversample): {oversampled_distribution}")
            print("Replication info per class:")
            # Correctly map label index to class name for printing
            idx_to_class = {s[2]: s[1] for s in samples}
            for label_idx, info in replication_info.items():
                 print(f"  {idx_to_class[label_idx]}: {info}")
        else:
            # If not oversampling, all are 'original'
            train_samples_to_write = [(idx, 'original') for idx in train_idx]
            print(f"Train samples: {len(train_idx)}")

        # Process training samples
        for idx, aug_type in train_samples_to_write:
            rows.append({
                "fold": fold,
                "split": "train",
                "class_name": class_names[idx],
                "image_path": image_paths[idx],
                "augmentation_type": aug_type
            })

        # Process test samples (no augmentation)
        for idx in test_idx:
            rows.append({
                "fold": fold,
                "split": "test",
                "class_name": class_names[idx],
                "image_path": image_paths[idx],
                "augmentation_type": 'none'
            })

    # The rest of the function for checks remains the same...

    with open(output_csv, "w", newline='') as f:
        # MODIFIED: Added 'augmentation_type' to fieldnames
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "class_name", "image_path", "augmentation_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved k-fold CSV with augmentation info to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--ignore_classes", nargs="+", default=None)
    parser.add_argument("--oversample", action="store_true", help="Oversample training set with augmentations")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.data_dir, "kfold_augmentations.csv")

    samples = collect_nested_samples(args.data_dir, args.ignore_classes)

    if not samples:
        print("❌ No valid image samples found.")
        exit(1)

    create_kfold_csv(samples, args.folds, args.output_csv, oversample=args.oversample)