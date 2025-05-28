#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import time
import copy
import csv
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torchvision import models # Import models for baseline
from tqdm import tqdm

# Import project modules
from datasets.custom_dataset import CustomDataset
from models.simclr_model import SimCLR
from models.byol_model import BYOL
from models.dino_model import DINO
from trainers.simclr_trainer import SimCLRTrainer
from trainers.byol_trainer import BYOLTrainer
from trainers.dino_trainer import DINOTrainer
from utils.logging_utils import log_message
from utils.metrics_utils import calculate_metrics, save_fold_results, save_aggregated_results
from utils.transforms import SimCLRTransform, BYOLTransform, DINOTransform, get_eval_transform

# --- Configuration ---
# General
DATASET_DIR = os.getenv("DATASET_DIR", "dataset-mestrado-Gabriel") # Default path, adjust if needed
RESULTS_DIR = os.getenv("RESULTS_DIR", "experiments/home/ubuntu/ssl_experiment/results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = int(os.getenv("NUM_FOLDS", 5))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))

# Model/Trainer Specific (using defaults from trainers, can be overridden via env vars)
BACKBONE = os.getenv("BACKBONE", "resnet18")
PRETRAIN_EPOCHS = int(os.getenv("PRETRAIN_EPOCHS", 50)) # Reduced for faster testing
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", 50)) # Reduced for faster testing
BASELINE_EPOCHS = int(os.getenv("BASELINE_EPOCHS", FINETUNE_EPOCHS)) # Baseline trains for same duration as finetuning
BASELINE_LR = float(os.getenv("BASELINE_LR", 1e-3))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

# Allowed classes (example, adjust as needed or set to None to use all)
# ALLOWED_CLASSES = ["class1", "class2", "class3"]
ALLOWED_CLASSES = None

# --- Helper Functions ---
def get_dataloaders(dataset, train_idx, test_idx, pretrain_transform, eval_transform):
    """Creates DataLoaders for a given fold."""
    # Create Subset datasets
    train_subset_pretrain = Subset(copy.deepcopy(dataset), train_idx)
    train_subset_finetune = Subset(copy.deepcopy(dataset), train_idx)
    test_subset = Subset(copy.deepcopy(dataset), test_idx)

    # Assign transforms
    if pretrain_transform:
        train_subset_pretrain.dataset.transform = pretrain_transform
    train_subset_finetune.dataset.transform = eval_transform
    test_subset.dataset.transform = eval_transform

    # Create DataLoaders
    pretrain_loader = DataLoader(train_subset_pretrain, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True) if pretrain_transform else None

    # Split fine-tuning/baseline training data for validation
    num_train_finetune = len(train_subset_finetune)
    indices = list(range(num_train_finetune))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * num_train_finetune)) # 10% for validation
    train_finetune_idx, val_finetune_idx = indices[split:], indices[:split]

    train_finetune_subset_actual = Subset(train_subset_finetune, train_finetune_idx)
    val_finetune_subset = Subset(train_subset_finetune, val_finetune_idx)

    finetune_train_loader = DataLoader(train_finetune_subset_actual, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    finetune_val_loader = DataLoader(val_finetune_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader

def run_supervised_training(train_loader, val_loader, test_loader, num_classes, class_names, fold_num, log_filepath, use_pretrained_weights, baseline_name):
    """Runs supervised training (either from scratch or fine-tuning ImageNet)."""
    log_message(log_filepath, f"\n=== Running {baseline_name} (Fold {fold_num+1}/{NUM_FOLDS}) ===")

    # 1. Instantiate backbone
    weights_arg = models.ResNet18_Weights.DEFAULT if use_pretrained_weights else None
    if BACKBONE == "resnet18":
        model = models.resnet18(weights=weights_arg)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif BACKBONE == "resnet50":
        weights_arg = models.ResNet50_Weights.DEFAULT if use_pretrained_weights else None
        model = models.resnet50(weights=weights_arg)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        log_message(log_filepath, f"Unsupported backbone {BACKBONE} for baseline. Exiting.")
        return {}

    log_message(log_filepath, f"Initializing {BACKBONE} with {'ImageNet weights' if use_pretrained_weights else 'random weights'} for {baseline_name}.")
    model = model.to(DEVICE)

    # 2. Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=BASELINE_LR)
    criterion = nn.CrossEntropyLoss()
    # Corrected StepLR step_size calculation
    step_size = max(1, BASELINE_EPOCHS // 3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    log_message(log_filepath, f"Using StepLR scheduler with step_size={step_size}")

    # 3. Training Loop with Validation and Early Stopping
    best_val_loss = float("inf")
    no_improve_counter = 0
    start_train = time.time()
    train_loss_history = []

    log_message(log_filepath, f"Starting {baseline_name} supervised training for {BASELINE_EPOCHS} epochs.")
    for epoch in range(BASELINE_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"{baseline_name} Epoch {epoch+1}/{BASELINE_EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
        train_loss_history.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0

        epoch_time = time.time() - epoch_start_time
        log_message(log_filepath, f"{baseline_name} Epoch {epoch+1}/{BASELINE_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

        scheduler.step()

        # Early stopping check
        if avg_val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val_loss
            no_improve_counter = 0
            # torch.save(model.state_dict(), f"{baseline_name}_best_fold_{fold_num+1}.pth") # Optional: save best model
        else:
            no_improve_counter += 1
            log_message(log_filepath, f"Early stopping counter: {no_improve_counter}/{EARLY_STOPPING_PATIENCE}")
            if no_improve_counter >= EARLY_STOPPING_PATIENCE:
                log_message(log_filepath, f"Early stopping triggered at epoch {epoch+1}")
                break

    train_time = time.time() - start_train
    log_message(log_filepath, f"{baseline_name} training finished. Total time: {train_time:.2f}s")

    # 4. Evaluation on Test Set
    model.eval()
    y_true = []
    y_pred = []
    start_eval = time.time()
    with torch.no_grad():
        pbar_eval = tqdm(test_loader, desc=f"Evaluating {baseline_name}", leave=False)
        for inputs, labels in pbar_eval:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    eval_time = time.time() - start_eval

    metrics = calculate_metrics(y_true, y_pred, num_classes, class_names)
    metrics["train_time"] = train_time
    metrics["eval_time"] = eval_time

    # Log final metrics
    acc = metrics["accuracy"]
    f1_macro = metrics["f1_macro"]
    log_message(log_filepath, f"{baseline_name} Fold {fold_num+1} Evaluation Results:")
    log_message(log_filepath, f"  Accuracy: {acc:.4f}")
    log_message(log_filepath, f"  F1 Macro: {f1_macro:.4f}")
    if class_names:
        for i, name in enumerate(class_names):
            metric_key = f"f1_{name}"
            f1_value = metrics.get(metric_key, "N/A")
            log_message(log_filepath, f"  F1 {name}: {f1_value if isinstance(f1_value, str) else f'{f1_value:.4f}'}")
    else:
         for i in range(num_classes):
            metric_key = f"f1_class_{i}"
            f1_value = metrics.get(metric_key, "N/A")
            log_message(log_filepath, f"  F1 Class {i}: {f1_value if isinstance(f1_value, str) else f'{f1_value:.4f}'}")
    log_message(log_filepath, f"{baseline_name} Fold {fold_num+1} Times - Train: {train_time:.2f}s, Eval: {eval_time:.2f}s")

    return metrics

def run_simclr(pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader, num_classes, class_names, fold_num, log_filepath):
    """Runs the SimCLR experiment for one fold."""
    log_message(log_filepath, f"\n=== Running SimCLR (Fold {fold_num+1}/{NUM_FOLDS}) ===")
    # SSL methods use pretrained ImageNet weights for backbone by default in their implementation
    model = SimCLR(backbone_name=BACKBONE, use_pretrained=True).to(DEVICE)
    trainer = SimCLRTrainer(model, device=DEVICE, log_filepath=log_filepath)

    start_pretrain = time.time()
    trainer.pretrain(pretrain_loader, epochs=PRETRAIN_EPOCHS)
    pretrain_time = time.time() - start_pretrain

    start_finetune = time.time()
    trainer.finetune(finetune_train_loader, finetune_val_loader, num_classes, epochs=FINETUNE_EPOCHS)
    finetune_time = time.time() - start_finetune

    start_eval = time.time()
    _, metrics = trainer.evaluate(test_loader, num_classes, class_names)
    eval_time = time.time() - start_eval

    metrics["pretrain_time"] = pretrain_time
    metrics["finetune_time"] = finetune_time
    metrics["eval_time"] = eval_time
    log_message(log_filepath, f"SimCLR Fold {fold_num+1} Times - Pretrain: {pretrain_time:.2f}s, Finetune: {finetune_time:.2f}s, Eval: {eval_time:.2f}s")
    return metrics

def run_byol(pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader, num_classes, class_names, fold_num, log_filepath):
    """Runs the BYOL experiment for one fold."""
    log_message(log_filepath, f"\n=== Running BYOL (Fold {fold_num+1}/{NUM_FOLDS}) ===")
    model = BYOL(backbone_name=BACKBONE, use_pretrained=True).to(DEVICE)
    trainer = BYOLTrainer(model, device=DEVICE, log_filepath=log_filepath)

    start_pretrain = time.time()
    trainer.pretrain(pretrain_loader, epochs=PRETRAIN_EPOCHS)
    pretrain_time = time.time() - start_pretrain

    start_finetune = time.time()
    trainer.finetune(finetune_train_loader, finetune_val_loader, num_classes, epochs=FINETUNE_EPOCHS)
    finetune_time = time.time() - start_finetune

    start_eval = time.time()
    _, metrics = trainer.evaluate(test_loader, num_classes, class_names)
    eval_time = time.time() - start_eval

    metrics["pretrain_time"] = pretrain_time
    metrics["finetune_time"] = finetune_time
    metrics["eval_time"] = eval_time
    log_message(log_filepath, f"BYOL Fold {fold_num+1} Times - Pretrain: {pretrain_time:.2f}s, Finetune: {finetune_time:.2f}s, Eval: {eval_time:.2f}s")
    return metrics

def run_dino(pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader, num_classes, class_names, fold_num, log_filepath):
    """Runs the DINO experiment for one fold."""
    log_message(log_filepath, f"\n=== Running DINO (Fold {fold_num+1}/{NUM_FOLDS}) ===")
    model = DINO(backbone_name=BACKBONE, use_pretrained=True).to(DEVICE)
    trainer = DINOTrainer(model, device=DEVICE, log_filepath=log_filepath)

    start_pretrain = time.time()
    trainer.pretrain(pretrain_loader, epochs=PRETRAIN_EPOCHS)
    pretrain_time = time.time() - start_pretrain

    start_finetune = time.time()
    trainer.finetune(finetune_train_loader, finetune_val_loader, num_classes, epochs=FINETUNE_EPOCHS)
    finetune_time = time.time() - start_finetune

    start_eval = time.time()
    _, metrics = trainer.evaluate(test_loader, num_classes, class_names)
    eval_time = time.time() - start_eval

    metrics["pretrain_time"] = pretrain_time
    metrics["finetune_time"] = finetune_time
    metrics["eval_time"] = eval_time
    log_message(log_filepath, f"DINO Fold {fold_num+1} Times - Pretrain: {pretrain_time:.2f}s, Finetune: {finetune_time:.2f}s, Eval: {eval_time:.2f}s")
    return metrics

# --- Main Execution --- 
def main():
    # Setup logging and results files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(RESULTS_DIR, f"experiment_log_{timestamp}.txt")
    fold_results_filepath = os.path.join(RESULTS_DIR, f"fold_results_{timestamp}.csv")
    agg_results_filepath = os.path.join(RESULTS_DIR, f"aggregated_results_{timestamp}.txt")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    log_message(log_filepath, "Starting SSL Comparison Experiment")
    log_message(log_filepath, f"Device: {DEVICE}")
    log_message(log_filepath, f"Dataset Dir: {DATASET_DIR}")
    log_message(log_filepath, f"Backbone: {BACKBONE}")
    log_message(log_filepath, f"Folds: {NUM_FOLDS}, Batch Size: {BATCH_SIZE}")
    log_message(log_filepath, f"Pretrain Epochs: {PRETRAIN_EPOCHS}, Finetune Epochs: {FINETUNE_EPOCHS}, Baseline Epochs: {BASELINE_EPOCHS}")

    # Load full dataset once
    try:
        full_dataset = CustomDataset(root_dir=DATASET_DIR, allowed_classes=ALLOWED_CLASSES)
        labels = full_dataset.labels
        num_classes = full_dataset.num_classes
        class_names = [full_dataset.idx_to_class[i] for i in range(num_classes)]
        log_message(log_filepath, f"Dataset loaded: {len(full_dataset)} samples, {num_classes} classes: {class_names}")
    except Exception as e:
        log_message(log_filepath, f"Error loading dataset: {e}")
        return

    # Prepare transforms
    simclr_pretrain_transform = SimCLRTransform(image_size=IMAGE_SIZE)
    byol_pretrain_transform = BYOLTransform(image_size=IMAGE_SIZE)
    dino_pretrain_transform = DINOTransform(image_size=IMAGE_SIZE)
    eval_transform = get_eval_transform(image_size=IMAGE_SIZE)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Updated to include both baselines
    all_fold_results = {"Baseline_Scratch": [], "Baseline_ImageNet": [], "SimCLR": [], "BYOL": [], "DINO": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        log_message(log_filepath, f"\n===== Starting Fold {fold+1}/{NUM_FOLDS} =====")
        fold_start_time = time.time()
        fold_results = {}

        # Get dataloaders for this fold (only need eval transform for baselines)
        _, ft_train_loader, ft_val_loader, test_loader_baseline = get_dataloaders(
            full_dataset, train_idx, test_idx, None, eval_transform
        )

        # --- Baseline (Scratch) ---
        try:
            baseline_scratch_metrics = run_supervised_training(ft_train_loader, ft_val_loader, test_loader_baseline, num_classes, class_names, fold, log_filepath, use_pretrained_weights=False, baseline_name="Baseline_Scratch")
            fold_results["Baseline_Scratch"] = baseline_scratch_metrics
            all_fold_results["Baseline_Scratch"].append(baseline_scratch_metrics)
        except Exception as e:
            log_message(log_filepath, f"Error running Baseline_Scratch in Fold {fold+1}: {e}")
            fold_results["Baseline_Scratch"] = {}

        # --- Baseline (ImageNet) ---
        try:
            # Need to re-get loaders if dataset state was modified, but should be ok here
            baseline_imagenet_metrics = run_supervised_training(ft_train_loader, ft_val_loader, test_loader_baseline, num_classes, class_names, fold, log_filepath, use_pretrained_weights=True, baseline_name="Baseline_ImageNet")
            fold_results["Baseline_ImageNet"] = baseline_imagenet_metrics
            all_fold_results["Baseline_ImageNet"].append(baseline_imagenet_metrics)
        except Exception as e:
            log_message(log_filepath, f"Error running Baseline_ImageNet in Fold {fold+1}: {e}")
            fold_results["Baseline_ImageNet"] = {}

        # --- SimCLR ---
        try:
            pt_loader_simclr, ft_train_loader_simclr, ft_val_loader_simclr, test_loader_simclr = get_dataloaders(
                full_dataset, train_idx, test_idx, simclr_pretrain_transform, eval_transform
            )
            simclr_metrics = run_simclr(pt_loader_simclr, ft_train_loader_simclr, ft_val_loader_simclr, test_loader_simclr, num_classes, class_names, fold, log_filepath)
            fold_results["SimCLR"] = simclr_metrics
            all_fold_results["SimCLR"].append(simclr_metrics)
        except Exception as e:
            log_message(log_filepath, f"Error running SimCLR in Fold {fold+1}: {e}")
            fold_results["SimCLR"] = {}

        # --- BYOL ---
        try:
            pt_loader_byol, ft_train_loader_byol, ft_val_loader_byol, test_loader_byol = get_dataloaders(
                full_dataset, train_idx, test_idx, byol_pretrain_transform, eval_transform
            )
            byol_metrics = run_byol(pt_loader_byol, ft_train_loader_byol, ft_val_loader_byol, test_loader_byol, num_classes, class_names, fold, log_filepath)
            fold_results["BYOL"] = byol_metrics
            all_fold_results["BYOL"].append(byol_metrics)
        except Exception as e:
            log_message(log_filepath, f"Error running BYOL in Fold {fold+1}: {e}")
            fold_results["BYOL"] = {}

        # --- DINO ---
        try:
            pt_loader_dino, ft_train_loader_dino, ft_val_loader_dino, test_loader_dino = get_dataloaders(
                full_dataset, train_idx, test_idx, dino_pretrain_transform, eval_transform
            )
            dino_metrics = run_dino(pt_loader_dino, ft_train_loader_dino, ft_val_loader_dino, test_loader_dino, num_classes, class_names, fold, log_filepath)
            fold_results["DINO"] = dino_metrics
            all_fold_results["DINO"].append(dino_metrics)
        except Exception as e:
            log_message(log_filepath, f"Error running DINO in Fold {fold+1}: {e}")
            fold_results["DINO"] = {}

        # Save results for the current fold
        save_fold_results(fold_results, fold + 1, fold_results_filepath, log_filepath)
        fold_time = time.time() - fold_start_time
        log_message(log_filepath, f"===== Fold {fold+1} finished. Time: {fold_time:.2f}s =====")

    # Aggregate results across folds
    log_message(log_filepath, "\nAggregating results across folds...")
    aggregated_results = {}
    metric_keys_to_aggregate = [] # Dynamically find keys like accuracy, f1_macro, f1_class_X, times

    # Determine keys from the first fold's results (use Baseline_Scratch or fallback)
    if all_fold_results["Baseline_Scratch"]: 
        first_result = all_fold_results["Baseline_Scratch"][0]
    elif all_fold_results["Baseline_ImageNet"]:
        first_result = all_fold_results["Baseline_ImageNet"][0]
    elif all_fold_results["SimCLR"]:
        first_result = all_fold_results["SimCLR"][0]
    else:
        first_result = None
        log_message(log_filepath, "Warning: No successful runs found to determine metric keys for aggregation.")

    if first_result:
        for key in first_result.keys():
            if key != "confusion_matrix": # Exclude non-scalar metrics for simple aggregation
                metric_keys_to_aggregate.append(key)

    for model_name, fold_data_list in all_fold_results.items():
        if not fold_data_list: # Skip if model failed in all folds
            log_message(log_filepath, f"No results to aggregate for {model_name}")
            continue

        aggregated_results[model_name] = {}
        for key in metric_keys_to_aggregate:
            # Collect values for this key across folds where the key exists
            values = [fold_data.get(key) for fold_data in fold_data_list if fold_data and key in fold_data]
            # Filter out None values if key was missing in some folds
            valid_values = [v for v in values if v is not None]
            if valid_values:
                aggregated_results[model_name][key] = valid_values # Store list of values
            else:
                 aggregated_results[model_name][key] = [] # Store empty list if no valid values found

    # Save aggregated results (mean/std calculated in the save function)
    save_aggregated_results(aggregated_results, agg_results_filepath, log_filepath)

    log_message(log_filepath, "Experiment finished.")

if __name__ == "__main__":
    # Ensure dataset directory exists (or provide instructions)
    if not os.path.isdir(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        print("Please set the DATASET_DIR environment variable or place the dataset there.")
        # Example: export DATASET_DIR=/path/to/your/dataset
    else:
        main()

