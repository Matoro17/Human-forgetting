import os
import csv
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from PIL import Image, UnidentifiedImageError
from collections import Counter


def collect_nested_samples(data_dir):
    samples = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
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


def create_kfold_csv(samples, k, output_csv):
    image_paths = [s[0] for s in samples]
    class_names = [s[1] for s in samples]
    labels = [s[2] for s in samples]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    rows = []

    all_test_paths_overall = set() # To ensure all test sets are distinct and cover the dataset
    test_fold_fingerprints = []    # To check for unique test sets per fold

    print(f"Total samples: {len(samples)}")
    print(f"Overall class distribution: {Counter(class_names)}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        print(f"Train samples in fold: {len(train_idx)}")
        print(f"Test samples in fold: {len(test_idx)}")

        current_test_paths = set()
        current_train_paths = set()

        # Collect paths and add to rows
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
        
        # --- Validation Checks ---

        # 1. Check if test sets are unique across folds (this is crucial for K-Fold)
        fingerprint = frozenset(current_test_paths)
        if fingerprint in test_fold_fingerprints:
            print(f"⚠️ WARNING: Test set for Fold {fold} is identical to a previous fold's test set!")
            print("This indicates a severe issue with StratifiedKFold, which should not happen.")
            print("Please check your data and scikit-learn version.")
        test_fold_fingerprints.append(fingerprint)
        
        # 2. Check for overlap between train and test sets within the current fold
        if not current_train_paths.isdisjoint(current_test_paths):
            print(f"❌ ERROR: Train and test sets for Fold {fold} overlap! This should never happen.")
            print(f"Overlapping samples: {current_train_paths.intersection(current_test_paths)}")
            exit(1) # Critical error, stop execution

        # 3. Check class distribution within this fold's train and test sets
        train_class_distribution = Counter(class_names[i] for i in train_idx)
        test_class_distribution = Counter(class_names[i] for i in test_idx)
        print(f"Train class distribution: {train_class_distribution}")
        print(f"Test class distribution: {test_class_distribution}")

        # 4. Add current test paths to the overall set for final coverage check
        all_test_paths_overall.update(current_test_paths)

    # --- Final Cross-Validation Summary ---
    print("\n--- Final Cross-Validation Summary ---")

    # 5. Ensure all original samples are covered exactly once by test sets
    if len(all_test_paths_overall) != len(image_paths):
        print(f"❌ ERROR: The total number of unique samples in all test sets ({len(all_test_paths_overall)}) "
              f"does not match the total number of original samples ({len(image_paths)}).")
        print("This means some samples were missed or duplicated in test sets across folds.")
    else:
        print(f"✅ All {len(image_paths)} samples were included exactly once in a test set across all folds.")

    with open(output_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "class_name", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved k-fold CSV to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to dataset (e.g., dataset-mestrado-Gabriel)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    # Set default output path inside data_dir if not provided
    if args.output_csv is None:
        args.output_csv = os.path.join(args.data_dir, "kfold_symlinks.csv")

    samples = collect_nested_samples(args.data_dir)

    if not samples:
        print("❌ No valid image samples found.")
        exit(1)

    create_kfold_csv(samples, args.folds, args.output_csv)