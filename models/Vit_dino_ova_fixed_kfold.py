import torch
import os
import sys
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from codecarbon import OfflineEmissionsTracker
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from dino import DINO, DINOTrainer, get_transform as dino_transform, log_message, save_metrics_to_txt
from vitdino import ViTDINO, ViTTrainer, get_transform as vit_transform
from datasets.symlinked_dataset import SymlinkedDataset
from utils.normalization import mean_std_for_symlinks  # Your normalization module
from utils.plot import plot_loss, plot_acc, plot_confusion_matrix  # Your visualization module

# Configuration
K_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))
CLASSES = [
    '0_Amiloidose', '1_Normal', '2_Esclerose_Pura_Sem_Crescente',
    '3_Hipercelularidade', '4_Hipercelularidade_Pura_Sem_Crescente',
    '5_Crescent', '6_Membranous', '7_Sclerosis', '8_Podocytopathy'
]

# Environment setup
load_dotenv()
BASE_DATA_DIR = os.getenv("DATASET_DIR", "/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/baseline/")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize logging
log_filename = f"DINO_ViT_OVA_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(RESULTS_DIR, log_filename)

class EnhancedTrainer:
    def __init__(self, architecture: str = 'dino_resnet18'):
        self.architecture = architecture
        self.input_size = (224, 224)
        self.results = {}
        self.normalization_metrics = {}

    def _get_normalization_stats(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and store normalization metrics"""
        mean, std = mean_std_for_symlinks(data_path, self.input_size)
        self.normalization_metrics[data_path] = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        return mean, std

    def _create_datasets(self, class_name: str, fold: int) -> Tuple[Dataset, Dataset]:
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')

        # Calculate and store normalization metrics
        train_mean, train_std = self._get_normalization_stats(train_dir)
        
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
        ])
        
        return (
            SymlinkedDataset(train_dir, transform=transform),
            SymlinkedDataset(val_dir, transform=transform)
        )

    def _run_fold(self, class_name: str, fold: int) -> Dict:
        log_message(log_filepath, f"\nStarting fold {fold+1} for {class_name}")
        
        # Initialize fresh model and trainer
        if self.architecture.startswith('vit'):
            model = ViTDINO(architecture=self.architecture)
            trainer = ViTTrainer(model, device=device)
        else:
            model = DINO(architecture=self.architecture.replace('dino_', ''))
            trainer = DINOTrainer(model, device=device)

        # Create datasets and loaders
        train_dataset, val_dataset = self._create_datasets(class_name, fold)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize tracking
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        # Train with metric collection
        train_metrics = {
            'pretrain_loss': [],
            'finetune_loss': [],
            'train_acc': []
        }

        # Pretraining phase
        trainer.train(train_loader, NUM_EPOCHS)
        train_metrics['pretrain_loss'] = trainer.loss_history
        
        # Fine-tuning phase
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)
        train_metrics['finetune_loss'] = trainer.loss_history
        train_metrics['train_acc'] = trainer.acc_history

        # Evaluation
        val_metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Save visualizations
        self._save_fold_visualizations(
            train_metrics, 
            val_metrics, 
            class_name, 
            fold
        )

        # Combine metrics
        return {
            **val_metrics,
            'training_time': total_time,
            'co2_emissions': emissions,
            'normalization_stats': self.normalization_metrics,
            'pretrain_loss_history': train_metrics['pretrain_loss'],
            'finetune_loss_history': train_metrics['finetune_loss'],
            'train_acc_history': train_metrics['train_acc']
        }

    def _save_fold_visualizations(self, train_metrics, val_metrics, class_name, fold):
        save_dir = os.path.join(RESULTS_DIR, self.architecture, class_name, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        # Loss curves
        plot_loss({
            'pretrain_loss': train_metrics['pretrain_loss'],
            'finetune_loss': train_metrics['finetune_loss'],
            'val_loss': val_metrics.get('loss_history', [])
        }, save_dir)

        # Accuracy curves
        plot_acc({
            'train_acc': train_metrics['train_acc'],
            'val_acc': val_metrics.get('acc_history', [])
        }, save_dir)

        # Confusion matrix
        plot_confusion_matrix(val_metrics['confusion_matrix'], save_dir)

        # Save normalization stats
        with open(os.path.join(save_dir, 'normalization.txt'), 'w') as f:
            f.write(f"Mean: {self.normalization_metrics['mean']}\n")
            f.write(f"Std: {self.normalization_metrics['std']}")

    def run_experiment(self):
        for class_name in CLASSES:
            class_metrics = []
            log_message(log_filepath, f"\n{'='*40}\nClass: {class_name}\n{'='*40}")
            
            for fold in range(K_FOLDS):
                fold_metrics = self._run_fold(class_name, fold)
                class_metrics.append(fold_metrics)
                log_message(log_filepath, f"Fold {fold+1} complete. Metrics: {fold_metrics}")

            # Aggregate class metrics
            self.results[class_name] = {
                'avg_f1': np.mean([m['f1_macro'] for m in class_metrics]),
                'std_dev': np.std([m['f1_macro'] for m in class_metrics]),
                'total_co2': sum([m['co2_emissions'] for m in class_metrics]),
                'normalization_stats': class_metrics[0]['normalization_stats']
            }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    experiments = {
        'dino_resnet50': EnhancedTrainer('dino_resnet50'),
        'vit_b_16': EnhancedTrainer('vit_b_16')
    }

    for name, experiment in experiments.items():
        log_message(log_filepath, f"\n{'#'*40}\nStarting {name} experiment\n{'#'*40}")
        experiment.run_experiment()
        save_metrics_to_txt(experiment.results, os.path.join(RESULTS_DIR, f"{name}_results.txt"))

    log_message(log_filepath, "\nAll experiments completed with full metrics collection")

if __name__ == "__main__":
    main()