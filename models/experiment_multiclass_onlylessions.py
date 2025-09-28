# experiment.py
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns

import timm
from torch.utils.data import WeightedRandomSampler
from collections import Counter

# Ensure custom modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datasets.custom_dataset_csv import CustomDatasetFromCSV
from dino import DINO, DINOTrainer, MultiCropTransform, log_message

# --- EXPERIMENT CONFIGURATION ---
CONFIG = {
    'DATA_DIR': '../../pathospotter/datasets/dataset-mestrado-Gabriel/', # <-- MUDE AQUI: Caminho para a pasta raiz dos seus dados
    'OUTPUT_DIR': './experiment_results_resnet_augmentations_dino', # Nova pasta de saída
    'CSV_PATH': '../../pathospotter/datasets/dataset-mestrado-Gabriel/kfold_augmentations.csv',   # <-- Novo: caminho para o CSV
    'ARCHITECTURE': 'resnet18',          # <-- Model to use (e.g., 'vit_base_patch16_224', 'swin_base_patch4_window7_224')
    'NUM_FOLDS': 5,
    'RANDOM_STATE': 42,
    'EPOCHS_PRETRAIN': 100,  # For real results, use 100+
    'EPOCHS_FINETUNE': 100,  # For real results, use 100+
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.0005, # Base LR for pre-training
    'CLASSES_TO_EXCLUDE': [],
    'TSNE_SAMPLES': 1000,
}

# --- HELPER FUNCTIONS ---

def generate_tsne_plot(model, dataloader, device, title, filename, num_samples, class_names):
    """Generates and saves a t-SNE plot of model features."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            # The dataloader for t-SNE uses the default transform (4D tensor)
            images = images.to(device)
            
            # **FIXED**: Get features from the backbone, not the final head
            features = model.backbone(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if len(np.concatenate(all_labels)) >= num_samples:
                break

    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    tsne = TSNE(n_components=2, random_state=CONFIG['RANDOM_STATE'], perplexity=min(30, len(all_features)-1))
    tsne_results = tsne.fit_transform(all_features)

    num_classes = len(class_names)
    palette = sns.color_palette("hsv", num_classes)

    plt.figure(figsize=(12, 8))
    for class_index in range(num_classes):
        indices = (all_labels == class_index)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    label=class_names[class_index],
                    alpha=0.7, s=40, color=palette[class_index])
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Classes")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def train_baseline_model(train_loader, test_loader, num_classes, epochs, device, log_filepath):
    """Trains and evaluates a model from scratch as a baseline."""
    log_message(log_filepath, f"\nStarting Baseline training ({CONFIG['ARCHITECTURE']} from scratch)")
    
    baseline_model = timm.create_model(
        CONFIG['ARCHITECTURE'], 
        pretrained=False, # From scratch
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    baseline_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    baseline_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = baseline_model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
    
    log_message(log_filepath, f"Baseline - F1-Micro: {f1_micro:.4f}")
    return {'f1_micro': f1_micro, 'f1_per_class': f1_per_class}

# --- MAIN EXPERIMENT ---

def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    LOG_FILEPATH = os.path.join(CONFIG['OUTPUT_DIR'], 'experiment.log')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    PRETRAINED_DIR = os.path.join(CONFIG['OUTPUT_DIR'], 'pretrained_weights')
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    
    log_message(LOG_FILEPATH, f"Starting new DINO experiment with Cross-Validation.")
    log_message(LOG_FILEPATH, f"Config: {CONFIG}")
    
    # Discover all unique classes from the CSV
    all_dataset_classes = set()
    with open(CONFIG['CSV_PATH'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['class_name'] not in CONFIG['CLASSES_TO_EXCLUDE']:
                all_dataset_classes.add(row['class_name'])
    
    class_names = sorted(list(all_dataset_classes))
    num_classes = len(class_names)
    
    log_message(LOG_FILEPATH, f"Classes used in experiment: {class_names}")
    
    dino_results = []
    baseline_results = []
    best_f1_micro = -1.0
    
    for fold in range(CONFIG['NUM_FOLDS']):
        log_message(LOG_FILEPATH, f"\n{'='*20} FOLD {fold}/{CONFIG['NUM_FOLDS']-1} {'='*20}")
        
        # --- Datasets and Dataloaders ---
        pretrain_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'], fold=fold, split='train',
            transform=MultiCropTransform(), classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR']
        )
        finetune_train_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'], fold=fold, split='train',
            transform=CustomDatasetFromCSV.get_default_transform(), classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR']
        )
        test_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'], fold=fold, split='test',
            transform=CustomDatasetFromCSV.get_default_transform(), classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR']
        )
        
        log_message(LOG_FILEPATH, f"Train samples: {len(finetune_train_dataset)}, Test samples: {len(test_dataset)}")
        
        # # Weighted sampler for imbalanced datasets during fine-tuning
        # train_labels = finetune_train_dataset.labels
        # class_counts = Counter(train_labels)
        # class_weights = {cls_idx: 1.0 / count for cls_idx, count in class_counts.items()}
        # sample_weights = [class_weights[label] for label in train_labels]
        # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        pretrain_loader = DataLoader(pretrain_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)
        
        # --- DINO Training ---
        dino_model = DINO(architecture=CONFIG['ARCHITECTURE'])
        dino_trainer = DINOTrainer(dino_model, device=DEVICE, base_lr=CONFIG['LEARNING_RATE'], log_filepath=LOG_FILEPATH)
        
        log_message(LOG_FILEPATH, f"Starting DINO pre-training for Fold {fold}")
        dino_trainer.train(pretrain_loader, num_epochs=CONFIG['EPOCHS_PRETRAIN'])
        
        torch.save(dino_trainer.model.state_dict(), os.path.join(PRETRAINED_DIR, f"dino_pretrained_fold_{fold}.pth"))
        
        log_message(LOG_FILEPATH, f"Starting DINO fine-tuning for Fold {fold}")
        dino_trainer.fine_tune(finetune_train_loader, num_classes=num_classes, epochs=CONFIG['EPOCHS_FINETUNE'])
        
        log_message(LOG_FILEPATH, f"Evaluating DINO model for Fold {fold}")
        fold_results = dino_trainer.evaluate(test_loader, num_classes=num_classes)
        dino_results.append(fold_results)

        if fold == 1:
            log_message(LOG_FILEPATH, "Generating t-SNE plot for Fold 1...")
            tsne_filename = os.path.join(CONFIG['OUTPUT_DIR'], f"tsne_dino_{CONFIG['ARCHITECTURE'].lower()}_finetuned.png")
            generate_tsne_plot(dino_trainer.model, test_loader, DEVICE, f"t-SNE (DINO-{CONFIG['ARCHITECTURE']} After Fine-Tuning)", tsne_filename, CONFIG['TSNE_SAMPLES'], class_names)
            log_message(LOG_FILEPATH, f"t-SNE plot saved to {tsne_filename}")

        if fold_results['f1_micro'] > best_f1_micro:
            best_f1_micro = fold_results['f1_micro']
            torch.save(dino_trainer.model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], f"best_dino_{CONFIG['ARCHITECTURE'].lower()}_model.pth"))
            log_message(LOG_FILEPATH, f"New best model saved with F1-Micro: {best_f1_micro:.4f}")

        # --- Baseline Training ---
        baseline_fold_results = train_baseline_model(finetune_train_loader, test_loader, num_classes, CONFIG['EPOCHS_FINETUNE'], DEVICE, LOG_FILEPATH)
        baseline_results.append(baseline_fold_results)
        
    # --- Final Analysis ---
    log_message(LOG_FILEPATH, f"\n\n{'='*20} FINAL RESULTS {'='*20}")
    
    dino_f1_micros = [r['f1_micro'] for r in dino_results]
    dino_f1_per_class = np.array([r['f1_per_class'] for r in dino_results])
    log_message(LOG_FILEPATH, f"\n--- DINO-{CONFIG['ARCHITECTURE']} (Pre-trained + Fine-tuning) ---")
    log_message(LOG_FILEPATH, f"Avg F1-Micro: {np.mean(dino_f1_micros):.4f} ± {np.std(dino_f1_micros):.4f}")
    mean_f1_per_class_dino = np.mean(dino_f1_per_class, axis=0)
    for i, c in enumerate(class_names):
        log_message(LOG_FILEPATH, f"  - Avg F1 '{c}': {mean_f1_per_class_dino[i]:.4f}")

    base_f1_micros = [r['f1_micro'] for r in baseline_results]
    log_message(LOG_FILEPATH, f"\n--- Baseline ({CONFIG['ARCHITECTURE']} from Scratch) ---")
    log_message(LOG_FILEPATH, f"Avg F1-Micro: {np.mean(base_f1_micros):.4f} ± {np.std(base_f1_micros):.4f}")

    # Wilcoxon Statistical Test
    log_message(LOG_FILEPATH, "\n--- Significance Test (Wilcoxon on F1-Micro) ---")
    try:
        stat, p_value = wilcoxon(dino_f1_micros, base_f1_micros)
        log_message(LOG_FILEPATH, f"Statistic: {stat:.4f}, P-value: {p_value:.4f}")
        if p_value < 0.05:
            log_message(LOG_FILEPATH, "Result: The difference is statistically significant.")
        else:
            log_message(LOG_FILEPATH, "Result: The difference is NOT statistically significant.")
    except ValueError as e:
        log_message(LOG_FILEPATH, f"Could not perform Wilcoxon test: {e}")

    log_message(LOG_FILEPATH, "\nExperiment concluded.")

if __name__ == '__main__':
    main()