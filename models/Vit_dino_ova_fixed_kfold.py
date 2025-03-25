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

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# Import components from existing files
from dino import DINO, DINOTrainer, get_transform as dino_transform, log_message, save_metrics_to_txt
from vitdino import ViTDINO, ViTTrainer, get_transform as vit_transform
from datasets.symlinked_dataset import SymlinkedDataset  # Ensure this is available

# Configuration
K_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50  # From .env or override
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

class OVAExperimentRunner:
    def __init__(self, architecture: str = 'dino_resnet18'):
        self.architecture = architecture
        self.transform = self._get_transform()
        self.results = {}

    def _get_transform(self):
        """Get appropriate transforms for architecture"""
        if self.architecture.startswith('vit'):
            return vit_transform(self.architecture)
        return dino_transform(self.architecture.replace('dino_', ''))

    def _initialize_model(self):
        """Initialize appropriate model architecture"""
        if self.architecture.startswith('vit'):
            return ViTDINO(architecture=self.architecture)
        return DINO(architecture=self.architecture.replace('dino_', ''))

    def _create_datasets(self, class_name: str, fold: int) -> tuple:
        """Create fold-specific datasets using symbolic links"""
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        return (
            SymlinkedDataset(os.path.join(fold_dir, 'train'), transform=self.transform),
            SymlinkedDataset(os.path.join(fold_dir, 'val'), transform=self.transform)
        )

    def _run_single_fold(self, class_name: str, fold: int) -> Dict:
        """Run complete training/evaluation for single fold"""
        log_message(log_filepath, f"\nStarting fold {fold+1} for {class_name}")
        
        # Initialize fresh model for each fold
        model = self._initialize_model()
        trainer_class = ViTTrainer if self.architecture.startswith('vit') else DINOTrainer
        trainer = trainer_class(model, device=device)

        # Create datasets
        train_dataset, val_dataset = self._create_datasets(class_name, fold)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training phase
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        trainer.train(train_loader, NUM_EPOCHS)
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)

        # Evaluation
        metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Add additional metrics
        metrics.update({
            'training_time': total_time,
            'co2_emissions': emissions,
            'class': class_name,
            'fold': fold+1
        })
        
        return metrics

    def run_class_experiment(self, class_name: str) -> Dict:
        """Run full K-fold experiment for a single class"""
        class_metrics = []
        log_message(log_filepath, f"\n{'='*40}\nStarting experiments for class: {class_name}\n{'='*40}")

        for fold in range(K_FOLDS):
            fold_metrics = self._run_single_fold(class_name, fold)
            class_metrics.append(fold_metrics)
            log_message(log_filepath, f"Fold {fold+1} metrics: {fold_metrics}")

        # Calculate average metrics across folds
        avg_metrics = {
            'f1_macro': np.mean([m['f1_macro'] for m in class_metrics]),
            'f1_positive': np.mean([m['f1_positive'] for m in class_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in class_metrics]),
            'co2_total': np.sum([m['co2_emissions'] for m in class_metrics]),
            'time_total': np.sum([m['training_time'] for m in class_metrics])
        }
        
        return avg_metrics

    def run_full_experiment(self):
        """Run OVA experiments for all classes"""
        for class_name in CLASSES:
            class_results = self.run_class_experiment(class_name)
            self.results[class_name] = class_results

def main():
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run experiments for different architectures
    architectures = ['dino_resnet18', 'vit_b_16']
    final_results = {}

    for arch in architectures:
        log_message(log_filepath, f"\n{'#'*40}\nStarting {arch} experiments\n{'#'*40}")
        runner = OVAExperimentRunner(architecture=arch)
        runner.run_full_experiment()
        final_results[arch] = runner.results

    # Save results
    save_metrics_to_txt(final_results, os.path.join(RESULTS_DIR, "ova_results_comparison.txt"))
    log_message(log_filepath, "\nAll experiments completed. Results saved.")

if __name__ == "__main__":
    main()