import torch
import os
import sys
import time
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

# Import only SimCLR components
from simclr import SimCLR, SimCLRTrainer, get_transform as simclr_transform
from datasets.symlinked_dataset import SymlinkedDataset
from utils.normalization import mean_std_for_symlinks
from utils.plot import plot_confusion_matrix
from dino import log_message, save_metrics_to_txt  # reuse utilities from dino

# Configuration
K_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
CLASSES = [
    '0_Amiloidose', '1_Normal', '2_Esclerose_Pura_Sem_Crescente',
    '3_Hipercelularidade', '4_Hipercelularidade_Pura_Sem_Crescente',
    '5_Crescent', '6_Membranous', '7_Sclerosis', '8_Podocytopathy'
]
ARCHITECTURES = ['simclr_resnet18']  # only SimCLR

# Environment setup
load_dotenv()
BASE_DATA_DIR = os.getenv("DATASET_DIR", "/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl/")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results_simclr_resnet18_5folds")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize logging
log_filename = f"SimCLR_Experiments_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(RESULTS_DIR, log_filename)


class UnifiedExperimentRunner:
    def __init__(self, architecture: str):
        self.architecture = architecture
        self.input_size = (224, 224)
        self.results = {}

    def _get_model(self):
        if self.architecture == 'simclr_resnet18':
            return SimCLR(backbone_name='resnet18')
        raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _get_trainer(self, model):
        return SimCLRTrainer(model, device=device)

    def _get_transform(self, mean, std):
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def _run_single_fold(self, class_name: str, fold: int) -> Dict:
        log_message(log_filepath, f"\nStarting {self.architecture} - {class_name} - Fold {fold + 1}")
        set_seeds(42)

        # Initialize model and trainer
        model = self._get_model()
        trainer = self._get_trainer(model)

        # Prepare dataset paths
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')

        # Calculate normalization stats
        train_mean, train_std = mean_std_for_symlinks(train_dir, self.input_size)
        transform = self._get_transform(train_mean, train_std)

        positive_classes = [f'1_{class_name.split("_", 1)[1]}']

        train_dataset = SymlinkedDataset(train_dir, transform=transform, binary_classification=True, positive_classes=positive_classes)
        val_dataset = SymlinkedDataset(val_dir, transform=transform, binary_classification=True, positive_classes=positive_classes)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(42))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        # Self-supervised pretraining
        trainer.train(train_loader, NUM_EPOCHS)

        # Supervised fine-tuning
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)

        metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Save results
        self._save_results(metrics, class_name, fold, total_time, emissions)

        return metrics

    def _save_results(self, metrics, class_name, fold, time_taken, emissions):
        save_dir = os.path.join(RESULTS_DIR, self.architecture, class_name, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Macro: {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Positive: {metrics['f1_positive']:.4f}\n")
            f.write(f"Class: {class_name}\n")
            f.write(f"Architecture: {self.architecture}\n")
            f.write(f"Fold: {fold}\n")
            f.write(f"Training Time: {time_taken:.2f}s\n")

        plot_confusion_matrix(metrics['confusion_matrix'], positive_class_name=class_name, save_path=save_dir)

    def run_class_experiment(self, class_name: str):
        class_metrics = []
        for fold in range(K_FOLDS):
            metrics = self._run_single_fold(class_name, fold)
            class_metrics.append(metrics)

        aggregated = {
            'avg_accuracy': np.mean([m['accuracy'] for m in class_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in class_metrics]),
            'avg_f1_macro': np.mean([m['f1_macro'] for m in class_metrics]),
            'std_f1_macro': np.std([m['f1_macro'] for m in class_metrics]),
            'avg_f1_positive': np.mean([m['f1_positive'] for m in class_metrics]),
            'std_f1_positive': np.std([m['f1_positive'] for m in class_metrics]),
        }

        self.results[class_name] = aggregated

    def run_full_experiment(self):
        for class_name in CLASSES:
            self.run_class_experiment(class_name)


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    final_results = {}

    for arch in ARCHITECTURES:
        log_message(log_filepath, f"\n{'#' * 40}\nStarting {arch} experiments\n{'#' * 40}")
        runner = UnifiedExperimentRunner(arch)
        runner.run_full_experiment()
        final_results[arch] = runner.results

    save_metrics_to_txt(final_results, os.path.join(RESULTS_DIR, "simclr_results.txt"))
    log_message(log_filepath, "\nAll SimCLR experiments completed successfully!")


if __name__ == "__main__":
    main()
