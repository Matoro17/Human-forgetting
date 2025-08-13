import os
import csv
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from PIL import Image, UnidentifiedImageError
from collections import Counter
import random


def collect_nested_samples(data_dir, ignore_classes=None):
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
    """Return new indices where minority classes are oversampled to match the max class count."""
    # Group indices by class
    class_to_indices = {}
    for idx in train_indices:
        label = labels[idx]
        class_to_indices.setdefault(label, []).append(idx)

    # Find max class size
    max_size = max(len(idxs) for idxs in class_to_indices.values())

    # Oversample
    new_train_indices = []
    replication_info = {}
    for label, idxs in class_to_indices.items():
        replication_factor = max_size // len(idxs)
        remainder = max_size % len(idxs)

        replicated = idxs * replication_factor + random.sample(idxs, remainder)
        new_train_indices.extend(replicated)

        replication_info[label] = {
            "original_count": len(idxs),
            "replicated_count": len(replicated),
            "factor": f"{replication_factor}x + {remainder} extra"
        }

    random.shuffle(new_train_indices)
    return new_train_indices, replication_info


def create_kfold_csv(samples, k, output_csv, oversample=False):
    image_paths = [s[0] for s in samples]
    class_names = [s[1] for s in samples]
    labels = [s[2] for s in samples]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    rows = []

    all_test_paths_overall = set()
    test_fold_fingerprints = []

    print(f"Total samples: {len(samples)}")
    print(f"Overall class distribution: {Counter(class_names)}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        print(f"Train samples before oversampling: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")

        train_class_distribution = Counter(class_names[i] for i in train_idx)
        test_class_distribution = Counter(class_names[i] for i in test_idx)
        print(f"Train class distribution (before oversample): {train_class_distribution}")
        print(f"Test class distribution: {test_class_distribution}")

        if oversample:
            train_idx, replication_info = oversample_train_set(train_idx, labels)
            oversampled_distribution = Counter(class_names[i] for i in train_idx)
            print(f"Train class distribution (AFTER oversample): {oversampled_distribution}")
            print("Replication info per class:")
            for lbl, info in replication_info.items():
                cname = samples[info['original_count'] and next(i for i, s in enumerate(samples) if s[2] == lbl)][1]
                print(f"  {cname}: {info}")

        current_test_paths = set()
        current_train_paths = set()

        for idx in train_idx:
            path = image_paths[idx]
            current_train_paths.add(path)
            rows.append({
                "fold": fold,
                "split": "train",
                "class_name": class_names[idx],
                "image_path": path
            })

        for idx in test_idx:
            path = image_paths[idx]
            current_test_paths.add(path)
            rows.append({
                "fold": fold,
                "split": "test",
                "class_name": class_names[idx],
                "image_path": path
            })

        # Checks
        fingerprint = frozenset(current_test_paths)
        if fingerprint in test_fold_fingerprints:
            print(f"⚠️ WARNING: Test set for Fold {fold} duplicates a previous fold's test set!")
        test_fold_fingerprints.append(fingerprint)

        if not current_train_paths.isdisjoint(current_test_paths):
            print(f"❌ ERROR: Train and test sets for Fold {fold} overlap!")
            exit(1)

        all_test_paths_overall.update(current_test_paths)

    if len(all_test_paths_overall) != len(image_paths):
        print(f"❌ ERROR: Test coverage mismatch. Found {len(all_test_paths_overall)} unique test samples.")
    else:
        print(f"✅ All samples are covered exactly once in test sets.")

    with open(output_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "class_name", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved k-fold CSV to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--ignore_classes", nargs="+", default=None,
                        help="List of class names to ignore")
    parser.add_argument("--oversample", action="store_true",
                        help="Oversample training set to balance classes")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.data_dir, "kfold_symlinks.csv")

    samples = collect_nested_samples(args.data_dir, args.ignore_classes)

    if not samples:
        print("❌ No valid image samples found.")
        exit(1)

    create_kfold_csv(samples, args.folds, args.output_csv, oversample=args.oversample)
