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

# Import all model implementations
from dino import DINO, DINOTrainer, log_message, save_metrics_to_txt
from dino import MultiCropTransform
from vitdino import ViTDINO, ViTTrainer, get_transform as vit_transform
from simclr import SimCLR, SimCLRTrainer, get_transform as simclr_transform
from datasets.symlinked_dataset import SymlinkedDataset
from utils.normalization import mean_std_for_symlinks
from utils.plot import plot_loss, plot_acc, plot_confusion_matrix

# Configuration
K_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))
CLASSES = [
    '0_Amiloidose','1_Normal','2_Esclerose_Pura_Sem_Crescente',
    '3_Hipercelularidade','4_Hipercelularidade_Pura_Sem_Crescente',
    '5_Crescent','6_Membranous','7_Sclerosis','8_Podocytopathy'
]
ARCHITECTURES = [
    # 'simclr_resnet18',
    'dino_resnet18'
]

# Environment setup
load_dotenv()
# This dataset doesn't have class balance
BASE_DATA_DIR = os.getenv("DATASET_DIR", "/home/alexsandro/pgcc/data/mestrado_Alexsandro/cross_validation/fsl/")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./resulst_dino_resnet18_5folds")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize logging
log_filename = f"SSL_Comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(RESULTS_DIR, log_filename)

class UnifiedExperimentRunner:
    def __init__(self, architecture: str):
        self.architecture = architecture
        self.input_size = self._get_input_size()
        self.results = {}
        self.normalization_metrics = {}

    def _get_input_size(self):
        if 'swin' in self.architecture:
            return (224, 224)  # Swin Transformers typically use 224x224
        return (224, 224)

    def _get_model(self):
        if self.architecture.startswith('simclr'):
            return SimCLR(backbone_name='resnet18')
        elif self.architecture.startswith('dino'):
            return DINO(architecture='resnet18', use_projection_head=True)
        elif self.architecture.startswith('vit'):
            return ViTDINO(architecture='vit_b_16')
        elif self.architecture.startswith('swin'):
            return self._create_swin_model()
        raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _create_swin_model(self):
        # Swin Transformer implementation similar to ViTDINO
        if self.architecture == 'swin_t':
            return ViTDINO(architecture='swin_t')
        elif self.architecture == 'swin_s':
            return ViTDINO(architecture='swin_s')
        elif self.architecture == 'swin_b':
            return ViTDINO(architecture='swin_b')
        raise ValueError(f"Unsupported Swin variant: {self.architecture}")

    def _get_trainer(self, model):
        if self.architecture.startswith('simclr'):
            return SimCLRTrainer(model, device=device)
        elif self.architecture.startswith('dino'):
            return DINOTrainer(model, device=device)
        elif self.architecture.startswith('vit') or self.architecture.startswith('swin'):
            return ViTTrainer(model, device=device)
        raise ValueError(f"Unsupported trainer for: {self.architecture}")

    def _get_transform(self, mean, std):
        if self.architecture.startswith('dino'):
            return MultiCropTransform()
        else:
            return transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def _run_single_fold(self, class_name: str, fold: int) -> Dict:
        log_message(log_filepath, f"\nStarting {self.architecture} - {class_name} - Fold {fold+1}")
        set_seeds(42)
        # Initialize fresh model and trainer
        model = self._get_model()
        trainer = self._get_trainer(model)

        # Create datasets with fold-specific normalization
        fold_dir = os.path.join(BASE_DATA_DIR, class_name, f"fold{fold}")
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')

        # Calculate normalization stats
        train_mean, train_std = mean_std_for_symlinks(train_dir, self.input_size)
        transform = self._get_transform(train_mean, train_std)
        
        # Dynamically determine class order based on main class name
        positive_classes = [f'1_{class_name.split("_", 1)[1]}'] 

        # Create datasets with explicit class order
        train_dataset = SymlinkedDataset(train_dir, transform=transform, binary_classification=True, positive_classes=positive_classes)
        val_dataset = SymlinkedDataset(val_dir, transform=transform, binary_classification=True, positive_classes=positive_classes)

        # Create loaders
        generator = torch.Generator().manual_seed(42)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator  # Fixed generator
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training and evaluation
        tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
        tracker.start()
        start_time = time.time()

        # Self-supervised pretraining
        trainer.train(train_loader, NUM_EPOCHS)
        
        # Supervised fine-tuning
        trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)
        
        # Evaluation
        # set_seeds(42)
        metrics = trainer.evaluate(val_loader)
        emissions = tracker.stop()
        total_time = time.time() - start_time

        # Save visualizations
        self._save_results(metrics, class_name, fold, total_time, emissions)

        return metrics

    def _save_results(self, metrics, class_name, fold, time, emissions):
        save_dir = os.path.join(RESULTS_DIR, self.architecture, class_name, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Macro: {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Positive: {metrics['f1_positive']:.4f}\n")
            f.write(f"Class: {class_name}\n")
            f.write(f"Architecture: {self.architecture}\n")
            f.write(f"Fold: {fold}\n")
            f.write(f"Training Time: {time:.2f}s\n")
            # f.write(f"CO2 Emissions: {emissions:.4f}kg\n")

        # Save confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], 
                     positive_class_name=class_name,  # This should be your positive class name
                     save_path=save_dir)

    def run_class_experiment(self, class_name: str):
        class_metrics = []
        for fold in range(K_FOLDS):
            fold_metrics = self._run_single_fold(class_name, fold)
            class_metrics.append(fold_metrics)
        
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
        log_message(log_filepath, f"\n{'#'*40}\nStarting {arch} experiments\n{'#'*40}")
        runner = UnifiedExperimentRunner(arch)
        runner.run_full_experiment()
        final_results[arch] = runner.results

    # Save consolidated results
    save_metrics_to_txt(final_results, os.path.join(RESULTS_DIR, "ssl_comparison_results.txt"))
    log_message(log_filepath, "\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()