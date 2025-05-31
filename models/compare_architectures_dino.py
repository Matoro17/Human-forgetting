#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import time
import copy
import csv
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# Import project modules
# Ensure these modules are accessible (e.g., in the same directory or PYTHONPATH)
try:
    from datasets.symlinked_dataset import SymlinkedDataset
except ImportError:
    print("Error: SymlinkedDataset.py not found. Make sure it's in the same directory or accessible via PYTHONPATH.")
    exit()
try:
    # Assuming dino_model_extended.py (or dino_updated_v2.py renamed to dino.py) contains DINO model, DINOTrainer, MultiCropTransform, log_message
    # Make sure to use the extended version that supports ViT/Swin
    from dino import DINO, DINOTrainer, MultiCropTransform, log_message # Assuming the updated model file is named dino.py
except ImportError as e:
    print(f"Error importing from dino: {e}. Make sure the updated DINO model file (dino_updated_v2.py) is saved as dino.py in the correct location.")
    exit()

# Import transforms (Corrected: Ensure this import is present)
try:
    from torchvision import transforms
except ImportError:
    print("Error: torchvision.transforms not found. Make sure torchvision is installed.")
    exit()

# --- Default Configuration --- (Can be overridden by args)
DEFAULT_K_FOLDS = int(os.getenv("K_FOLDS", 5))
DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
DEFAULT_NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
DEFAULT_PRETRAIN_EPOCHS = int(os.getenv("DINO_PRETRAIN_EPOCHS", 50))
DEFAULT_FINETUNE_EPOCHS = int(os.getenv("DINO_FINETUNE_EPOCHS", 50))
DEFAULT_EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
DEFAULT_EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))
DEFAULT_DATA_DIR = os.getenv("DATASET_DIR", "/home/ubuntu/symlink_dataset")
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", "./results_dino_multi_arch_comparison")

# List of binary classification tasks to run (based on top-level folder names in data_dir)
BINARY_TASKS = [
    "0_Amiloidose"
    # , "1_Normal", "2_Esclerose_Pura_Sem_Crescente",
    # "3_Hipercelularidade", "4_Hipercelularidade_Pura_Sem_Crescente",
    # "5_Crescent", "6_Membranous", "7_Sclerosis", "8_Podocytopathy"
]

# --- Architectures to Compare ---
ARCHITECTURES_TO_COMPARE = [
    'resnet18',
    'resnet50',
    'resnet101',
    'vit_b_16',
    'swin_t',
    'swin_s',
    'swin_b'
]

# --- Helper Functions ---
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# --- Main Experiment Class ---
class ArchitectureComparatorDINO:
    def __init__(self, architecture: str, data_dir: str, results_dir: str, log_filepath: str,
                 k_folds: int, random_state: int, batch_size: int, image_size: int, num_workers: int,
                 pretrain_epochs: int, finetune_epochs: int, early_stopping_patience: int, early_stopping_delta: float,
                 device: str):
        self.architecture = architecture
        self.data_dir = data_dir
        self.results_dir = results_dir # Base results dir
        self.arch_results_dir = os.path.join(results_dir, self.architecture)
        os.makedirs(self.arch_results_dir, exist_ok=True)
        self.log_filepath = log_filepath
        self.results = {} # Stores aggregated results for this architecture

        # Store config parameters passed from main
        self.k_folds = k_folds
        self.random_state = random_state
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.device = device

        log_message(self.log_filepath, f"\n===== Initializing Comparator for DINO with backbone: {self.architecture} =====")
        log_message(self.log_filepath, f"Data Dir: {self.data_dir}")
        log_message(self.log_filepath, f"Results Dir for this arch: {self.arch_results_dir}")

    def _get_model(self):
        """Instantiates the DINO model with the specified backbone."""
        try:
            model = DINO(architecture=self.architecture, use_pretrained=True)
            log_message(self.log_filepath, f"DINO model with {self.architecture} backbone created.")
            return model
        except ValueError as e:
            log_message(self.log_filepath, f"Error creating DINO model (ValueError): {e}. Is backbone '{self.architecture}' supported by your DINO class?")
            raise
        except Exception as e:
            log_message(self.log_filepath, f"Error creating DINO model (Exception): {e}")
            raise

    def _get_trainer(self, model):
        """Instantiates the DINOTrainer."""
        return DINOTrainer(model, device=self.device, log_filepath=self.log_filepath,
                           early_stopping_patience=self.early_stopping_patience,
                           early_stopping_delta=self.early_stopping_delta)

    def _get_transform(self, pretrain=True):
        """Returns the appropriate transform (MultiCrop for pretrain, Eval for finetune)."""
        if pretrain:
            # Corrected: Use 'num_local' instead of 'num_local_crops'
            return MultiCropTransform(global_crops_scale=(0.4, 1.0),
                                      local_crops_scale=(0.05, 0.4),
                                      num_local=6) # DINO default
        else:
            # Standard eval transform using torchvision.transforms
            return transforms.Compose([
                transforms.Resize(int(self.image_size * 1.14)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
            ])

    def _run_single_fold(self, task_name: str, fold: int):
        """Runs DINO pre-training and binary fine-tuning for a specific task and fold."""
        log_message(self.log_filepath, f"\n--- Starting Task: {task_name} | Fold: {fold+1}/{self.k_folds} | Arch: {self.architecture} ---")
        set_seeds(self.random_state + fold)

        try:
            model = self._get_model().to(self.device)
            trainer = self._get_trainer(model)
        except Exception as e:
            log_message(self.log_filepath, f"Error initializing model/trainer for {self.architecture}, fold {fold+1}: {e}")
            return {}

        fold_dir = os.path.join(self.data_dir, task_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")

        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            log_message(self.log_filepath, f"Error: Train ('{train_dir}') or Val ('{val_dir}') dir not found for task {task_name}, fold {fold+1}. Skipping fold.")
            return {}

        try:
            pretrain_transform = self._get_transform(pretrain=True)
            eval_transform = self._get_transform(pretrain=False)
            positive_classes = [f'1_{task_name.split("_", 1)[1]}'] 
            train_dataset_pretrain = SymlinkedDataset(train_dir, transform=pretrain_transform, binary_classification=True, positive_classes=positive_classes)
            train_dataset_finetune = SymlinkedDataset(train_dir, transform=eval_transform, binary_classification=True, positive_classes=positive_classes)
            val_dataset_eval = SymlinkedDataset(val_dir, transform=eval_transform, binary_classification=True, positive_classes=positive_classes)

            if not train_dataset_pretrain or not train_dataset_finetune or not val_dataset_eval:
                 log_message(self.log_filepath, f"Error: Empty dataset(s) created for task {task_name}, fold {fold+1}. Skipping fold.")
                 return {}

            generator = torch.Generator().manual_seed(self.random_state + fold)
            # Use parameters passed during init
            pretrain_loader = DataLoader(train_dataset_pretrain, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, generator=generator, drop_last=True)
            finetune_loader = DataLoader(train_dataset_finetune, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, generator=generator)
            eval_loader = DataLoader(val_dataset_eval, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        except Exception as e:
            log_message(self.log_filepath, f"Error creating datasets/loaders for fold {fold+1}: {e}")
            log_message(self.log_filepath, traceback.format_exc())
            return {}

        fold_results = {}
        try:
            # Use parameters passed during init
            log_message(self.log_filepath, f"Starting DINO Pre-training ({self.pretrain_epochs} epochs)...")
            start_pretrain = time.time()
            if hasattr(trainer, "pretrain"):
                trainer.pretrain(pretrain_loader, epochs=self.pretrain_epochs)
            elif hasattr(trainer, "train") and isinstance(trainer, DINOTrainer):
                 trainer.train(pretrain_loader, epochs=self.pretrain_epochs)
            else:
                 raise NotImplementedError("Trainer pretrain/train method missing")
            fold_results["pretrain_time_s"] = time.time() - start_pretrain
            log_message(self.log_filepath, f"Pre-training finished in {fold_results['pretrain_time_s']:.2f}s")

            log_message(self.log_filepath, f"Starting Binary Fine-tuning ({self.finetune_epochs} epochs)...")
            start_finetune = time.time()
            if hasattr(trainer, "finetune"):
                 trainer.finetune(finetune_loader, eval_loader, num_classes=2, epochs=self.finetune_epochs)
            else:
                 raise NotImplementedError("Trainer finetune method missing")
            fold_results["finetune_time_s"] = time.time() - start_finetune
            log_message(self.log_filepath, f"Fine-tuning finished in {fold_results['finetune_time_s']:.2f}s")

            log_message(self.log_filepath, "Starting Evaluation...")
            start_eval = time.time()
            _, metrics = trainer.evaluate(eval_loader, num_classes=2, class_names=["Negative", "Positive"])
            fold_results["eval_time_s"] = time.time() - start_eval
            fold_results.update(metrics)
            # Corrected quote usage in f-string
            log_message(self.log_filepath, f"Evaluation finished in {fold_results['eval_time_s']:.2f}s. Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

            self._save_fold_results(fold_results, task_name, fold)

        except Exception as e:
            log_message(self.log_filepath, f"Error during training/evaluation for task {task_name}, fold {fold+1}: {e}")
            log_message(self.log_filepath, traceback.format_exc())
            return {}

        log_message(self.log_filepath, f"--- Finished Task: {task_name} | Fold: {fold+1} ---")
        return fold_results

    def _save_fold_results(self, metrics, task_name, fold):
        """Saves metrics and plots for a single fold into the architecture's result folder."""
        save_dir = os.path.join(self.arch_results_dir, task_name, f"fold{fold+1}")
        os.makedirs(save_dir, exist_ok=True)

        metrics_filepath = os.path.join(save_dir, "metrics.txt")
        with open(metrics_filepath, "w") as f:
            f.write(f"Architecture: {self.architecture}\n")
            f.write(f"Task: {task_name}\n")
            f.write(f"Fold: {fold+1}\n")
            for key, value in metrics.items():
                if key != "confusion_matrix":
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

        if "confusion_matrix" in metrics:
            try:
                # Assuming plot_confusion_matrix is available
                from utils.plot import plot_confusion_matrix
                plot_confusion_matrix(metrics["confusion_matrix"],
                                      class_names=["Negative", "Positive"],
                                      save_path=save_dir)
            except ImportError:
                log_message(self.log_filepath, "Skipping confusion matrix plot: utils.plot not found or plot_confusion_matrix missing.")
            except Exception as e:
                log_message(self.log_filepath, f"Error saving confusion matrix plot: {e}")

    def run_task_experiment(self, task_name: str):
        """Runs the experiment for all folds of a specific task for the current architecture."""
        task_metrics_list = []
        log_message(self.log_filepath, f"\n===== Starting Task Experiment: {task_name} | Arch: {self.architecture} =====")
        for fold in range(self.k_folds):
            fold_metrics = self._run_single_fold(task_name, fold)
            if fold_metrics:
                task_metrics_list.append(fold_metrics)
            else:
                log_message(self.log_filepath, f"Fold {fold+1} for task {task_name} failed or was skipped.")

        if task_metrics_list:
            aggregated = self._aggregate_metrics(task_metrics_list)
            self.results[task_name] = aggregated
            log_message(self.log_filepath, f"Aggregated results for task {task_name} | Arch: {self.architecture}:")
            for key, (mean, std) in aggregated.items():
                log_message(self.log_filepath, f"  {key}: {mean:.4f} +/- {std:.4f}")
        else:
             log_message(self.log_filepath, f"No successful folds to aggregate for task {task_name} | Arch: {self.architecture}.")
             self.results[task_name] = {}
        log_message(self.log_filepath, f"===== Finished Task Experiment: {task_name} | Arch: {self.architecture} =====")
        return self.results

    def _aggregate_metrics(self, metrics_list):
        """Calculates mean and std dev for metrics across folds."""
        aggregated = {}
        if not metrics_list:
            return aggregated
        keys_to_aggregate = [k for k, v in metrics_list[0].items() if isinstance(v, (int, float))]
        for key in keys_to_aggregate:
            values = [m.get(key) for m in metrics_list if m and key in m and isinstance(m.get(key), (int, float))]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                aggregated[key] = (mean, std)
        return aggregated

    def save_aggregated_results_to_txt(self, filepath):
         """Saves the aggregated (mean +/- std) results for the current architecture to a text file."""
         os.makedirs(os.path.dirname(filepath), exist_ok=True)
         with open(filepath, "w") as f:
             f.write(f"Aggregated Results for DINO with {self.architecture} Backbone\n")
             f.write(f"Data Directory: {self.data_dir}\n")
             f.write(f"Number of Folds: {self.k_folds}\n") # Use stored k_folds
             f.write("--------------------------------------------------\n")
             for task_name, aggregated_metrics in self.results.items():
                 if not aggregated_metrics:
                     f.write(f"\nTask: {task_name}\n  No successful folds.\n")
                     continue
                 f.write(f"\nTask: {task_name}\n")
                 for key, (mean, std) in aggregated_metrics.items():
                     f.write(f"  {key}: {mean:.4f} +/- {std:.4f}\n")
                 f.write("--------------------------------------------------\n")
         log_message(self.log_filepath, f"Aggregated results for {self.architecture} saved to {filepath}")

# --- Main Execution --- 
def main():
    # Use default values defined at the top
    parser = argparse.ArgumentParser(description="Compare Multiple Architectures using DINO SSL")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Root directory containing task folders, each with fold subdirectories.")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save results and logs.")
    parser.add_argument("--k_folds", type=int, default=DEFAULT_K_FOLDS, help="Number of cross-validation folds.")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="Random state for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training and evaluation.")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Image size for transforms.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of workers for DataLoader.")
    parser.add_argument("--pretrain_epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS, help="Number of epochs for DINO pre-training.")
    parser.add_argument("--finetune_epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS, help="Number of epochs for fine-tuning.")
    parser.add_argument("--early_stopping_patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="Patience for early stopping.")
    parser.add_argument("--early_stopping_delta", type=float, default=DEFAULT_EARLY_STOPPING_DELTA, help="Minimum change to qualify as improvement for early stopping.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")

    args = parser.parse_args()

    # Create base results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Setup main log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_log_multi_arch_{timestamp}.txt"
    log_filepath = os.path.join(args.results_dir, log_filename)

    log_message(log_filepath, "Starting Multi-Architecture DINO Comparison Experiment")
    log_message(log_filepath, f"Arguments: {vars(args)}")
    log_message(log_filepath, f"Architectures to compare: {ARCHITECTURES_TO_COMPARE}")
    log_message(log_filepath, f"Binary Tasks: {BINARY_TASKS}")

    all_results = {} # Dictionary to store results across all architectures

    # Loop through each architecture
    for arch in ARCHITECTURES_TO_COMPARE:
        log_message(log_filepath, f"\n{'='*20} Starting Experiment for Architecture: {arch} {'='*20}")
        try:
            comparator = ArchitectureComparatorDINO(
                architecture=arch,
                data_dir=args.data_dir,
                results_dir=args.results_dir, # Pass base results dir
                log_filepath=log_filepath,
                k_folds=args.k_folds,
                random_state=args.random_state,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_delta=args.early_stopping_delta,
                device=args.device
            )

            arch_results = {} # Results for the current architecture
            # Loop through each binary task
            for task in BINARY_TASKS:
                task_results = comparator.run_task_experiment(task)
                arch_results.update(task_results) # Update results for this arch

            # Save aggregated results for the current architecture
            arch_summary_filepath = os.path.join(comparator.arch_results_dir, "aggregated_results.txt")
            comparator.save_aggregated_results_to_txt(arch_summary_filepath)
            all_results[arch] = comparator.results # Store results for this arch in the main dict

        except Exception as e:
            log_message(log_filepath, f"Critical error during experiment for architecture {arch}: {e}")
            log_message(log_filepath, traceback.format_exc())
            log_message(log_filepath, f"Skipping remaining tasks for architecture {arch}.")
            all_results[arch] = {"error": str(e)} # Mark architecture as failed

        log_message(log_filepath, f"{'='*20} Finished Experiment for Architecture: {arch} {'='*20}\n")

    # Save final summary across all architectures (Optional, could be complex)
    # For now, individual architecture summaries are saved.
    log_message(log_filepath, "\nMulti-Architecture DINO Comparison Experiment Finished.")
    log_message(log_filepath, f"Results saved in subdirectories within: {args.results_dir}")

if __name__ == "__main__":
    main()

