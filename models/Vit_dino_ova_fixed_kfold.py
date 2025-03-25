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
from typing import Dict, List
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Import components from existing files
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

class EnhancedOVAExperimentRunner:
    def __init__(self, architecture: str = 'dino_resnet18'):
        self.architecture = architecture
        self.results = {}
        self.input_size = (224, 224) if 'vit' not in architecture else (224, 224)

    def _get_transform(self, mean, std):
        """Get normalized transforms using calculated statistics"""
        base_transforms = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        return transforms.Compose(base_transforms)

    def _initialize_model(self):
        if self.architecture.startswith('vit'):
            return ViTDINO(architecture=self.architecture)
        return DINO(architecture=self.architecture.replace('dino_', ''))

    def _create_datasets(self, class_name: str, fold: int) -> tuple:
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')

        # Calculate normalization parameters
        train_mean, train_std = mean_std_for_symlinks(train_dir, self.input_size)
        
        return (
            SymlinkedDataset(train_dir, transform=self._get_transform(train_mean, train_std)),
            SymlinkedDataset(val_dir, transform=self._get_transform(train_mean, train_std))
        )

    def _run_single_fold(self, class_name: str, fold: int) -> Dict:
        log_message(log_filepath, f"\nStarting fold {fold+1} for {class_name}")
        
        model = self._initialize_model()
        trainer_class = ViTTrainer if self.architecture.startswith('vit') else DINOTrainer
        trainer = trainer_class(model, device=device)

        train_dataset, val_dataset = self._create_datasets(class_name, fold)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training with metrics collection
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        # Modified train method to return training metrics
        train_results = trainer.train(train_loader, NUM_EPOCHS)
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)

        # Evaluation
        metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Visualization
        self._save_visualizations(train_results, metrics, class_name, fold)

        metrics.update({
            'training_time': total_time,
            'co2_emissions': emissions,
            'class': class_name,
            'fold': fold+1
        })
        
        return metrics

    def _save_visualizations(self, train_results, val_metrics, class_name, fold):
        """Save training curves and confusion matrices"""
        save_dir = os.path.join(RESULTS_DIR, self.architecture, class_name, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        # Plot training curves
        plot_loss({
            'train_loss': train_results['train_losses'],
            'test_loss': val_metrics['loss_history']
        }, save_dir + '/')
        
        plot_acc({
            'train_acc': train_results['train_accuracies'],
            'test_acc': val_metrics['acc_history']
        }, save_dir + '/')

        # Plot confusion matrix
        plot_confusion_matrix(val_metrics['confusion_matrix'], save_dir + '/')

    def run_class_experiment(self, class_name: str) -> Dict:
        class_metrics = []
        log_message(log_filepath, f"\n{'='*40}\nStarting experiments for class: {class_name}\n{'='*40}")

        for fold in range(K_FOLDS):
            fold_metrics = self._run_single_fold(class_name, fold)
            class_metrics.append(fold_metrics)
            log_message(log_filepath, f"Fold {fold+1} metrics: {fold_metrics}")

        avg_metrics = {
            'f1_macro': np.mean([m['f1_macro'] for m in class_metrics]),
            'f1_positive': np.mean([m['f1_positive'] for m in class_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in class_metrics]),
            'co2_total': np.sum([m['co2_emissions'] for m in class_metrics]),
            'time_total': np.sum([m['training_time'] for m in class_metrics]),
            'confusion_matrices': [m['confusion_matrix'] for m in class_metrics]
        }
        
        return avg_metrics

    def run_full_experiment(self):
        for class_name in CLASSES:
            class_results = self.run_class_experiment(class_name)
            self.results[class_name] = class_results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    architectures = ['dino_resnet18', 'vit_b_16']
    final_results = {}

    for arch in architectures:
        log_message(log_filepath, f"\n{'#'*40}\nStarting {arch} experiments\n{'#'*40}")
        runner = EnhancedOVAExperimentRunner(architecture=arch)
        runner.run_full_experiment()
        final_results[arch] = runner.results

    save_metrics_to_txt(final_results, os.path.join(RESULTS_DIR, "enhanced_ova_results.txt"))
    log_message(log_filepath, "\nAll experiments completed with visualizations.")

if __name__ == "__main__":
    main()