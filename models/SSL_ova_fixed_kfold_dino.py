import torch
import os
import sys
import time
import math
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from codecarbon import OfflineEmissionsTracker
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dino import DINO, DINOTrainer, MultiCropTransform, log_message, save_metrics_to_txt
from datasets.symlinked_dataset import SymlinkedDataset
from utils.plot import plot_confusion_matrix

# Configuration
K_FOLDS = 1
BATCH_SIZE = 32
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
CLASSES = ['0_Amiloidose']
ARCHITECTURES = ['dino_resnet18']

# Environment setup
load_dotenv()
BASE_DATA_DIR = os.getenv("DATASET_DIR", "/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl/")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results_dino")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize logging
log_filename = f"DINO_Experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(RESULTS_DIR, log_filename)

class DINOExperimentRunner:
    def __init__(self):
        self.input_size = (224, 224)
        self.results = {}

    def _get_transform(self):
        """Fixed DINO transforms with ImageNet normalization"""
        return MultiCropTransform(
            global_size=224,
            local_size=96,
            num_local=4
        )

    def _collate_fn(self, batch):
        """Custom collate to handle multi-crop outputs"""
        images, labels = zip(*batch)
        
        # Stack all crops for each view type
        if isinstance(images[0], list):
            num_crops = len(images[0])
            return [torch.stack([imgs[i] for imgs in images]) for i in range(num_crops)], torch.tensor(labels)
        return torch.stack(images), torch.tensor(labels)

    def _run_single_fold(self, class_name: str, fold: int) -> Dict:
        log_message(log_filepath, f"\nStarting DINO - {class_name} - Fold {fold+1}")
        torch.manual_seed(42)

        # Initialize fresh model and trainer
        model = DINO(architecture='resnet18')
        trainer = DINOTrainer(model, device=device)

        # Create datasets with DINO-specific transform
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')

        transform = self._get_transform()
        positive_classes = [f'1_{class_name.split("_", 1)[1]}']

        # Create datasets
        train_dataset = SymlinkedDataset(
            train_dir,
            transform=transform,
            binary_classification=True,
            positive_classes=positive_classes
        )
        val_dataset = SymlinkedDataset(
            val_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            binary_classification=True,
            positive_classes=positive_classes
        )

        # Create loaders with custom collate
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Training and evaluation
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        # Self-supervised pretraining
        trainer.train(train_loader, NUM_EPOCHS)
        
        # Supervised fine-tuning
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)
        
        # Evaluation
        metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Save results
        self._save_results(metrics, class_name, fold, total_time, emissions)
        return metrics

    def _save_results(self, metrics, class_name, fold, time, emissions):
        save_dir = os.path.join(RESULTS_DIR, 'dino_resnet18', class_name, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Class: {class_name}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Training Time: {time:.2f}s\n")
            # f.write(f"CO2 Emissions: {emissions:.4f}kg\n")

        # Save confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], 
                            class_name, 
                            save_path=save_dir)

    def run_experiment(self):
        for class_name in CLASSES:
            class_metrics = []
            for fold in range(K_FOLDS):
                fold_metrics = self._run_single_fold(class_name, fold)
                class_metrics.append(fold_metrics)
            
            self.results[class_name] = {
                'avg_accuracy': np.mean([m['accuracy'] for m in class_metrics]),
                'avg_f1': np.mean([m['f1'] for m in class_metrics])
            }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runner = DINOExperimentRunner()
    runner.run_experiment()
    save_metrics_to_txt(runner.results, os.path.join(RESULTS_DIR, "dino_results.txt"))
    log_message(log_filepath, "\nExperiment completed!")

if __name__ == "__main__":
    main()