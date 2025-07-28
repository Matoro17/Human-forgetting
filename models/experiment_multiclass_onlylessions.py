# experiment.py
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold # Keep for conceptual understanding if needed, but not directly used for splitting here
from sklearn.manifold import TSNE
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import csv # Import csv
import seaborn as sns
# Adicione a importação do timm
import timm
from torch.utils.data import WeightedRandomSampler
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datasets.custom_dataset_csv import CustomDatasetFromCSV  # novo import
# Assumindo que as classes DINO e CustomDataset estão nos arquivos abaixo
from dino import DINO, DINOTrainer, MultiCropTransform, log_message
# from datasets.custom_dataset import CustomDataset # No longer needed

# --- CONFIGURAÇÕES DO EXPERIMENTO ATUALIZADAS ---
CONFIG = {
    'DATA_DIR': './dataset-mestrado-Gabriel', # <-- MUDE AQUI: Caminho para a pasta raiz dos seus dados
    'OUTPUT_DIR': './experiment_results_vit', # Nova pasta de saída
    'CSV_PATH': './dataset-mestrado-Gabriel/kfold_symlinks.csv',   # <-- Novo: caminho para o CSV
    'ARCHITECTURE': 'resnet18', # <-- ARQUITETURA BASE (DINO e Baseline)
    'NUM_FOLDS': 5,
    'RANDOM_STATE': 42,
    'EPOCHS_PRETRAIN': 4, # Aumente para resultados melhores (e.g., 100+)
    'EPOCHS_FINETUNE': 4,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.0005,
    'CLASSES_TO_EXCLUDE': ['1_Normal'], # <-- Classe a ser ignorada em TODAS as fases
    'TSNE_SAMPLES': 1000,
}

# --- FUNÇÕES AUXILIARES ---

def generate_tsne_plot(model, dataloader, device, title, filename, num_samples, class_names):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if isinstance(images, list):
                images = images[0]
            elif images.dim() == 5:
                images = images[:, 0]
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if len(np.concatenate(all_labels)) >= num_samples:
                break

    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    tsne = TSNE(n_components=2, random_state=CONFIG['RANDOM_STATE'], perplexity=min(30, len(all_features)-1))
    tsne_results = tsne.fit_transform(all_features)

    num_classes = len(class_names)

    # Gerar paleta qualitativa distinta
    palette = sns.color_palette("hsv", num_classes)   # paleta HSV espaçada

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

# --- ALTERADO: Função de baseline agora é genérica ---
def train_baseline_model(train_loader, test_loader, num_classes, epochs, device, log_filepath):
    """Treina e avalia um modelo do zero como baseline, usando a arquitetura da CONFIG."""
    log_message(log_filepath, f"\nIniciando treinamento do modelo Baseline ({CONFIG['ARCHITECTURE']} from scratch)")
    
    # Modelo baseline sem pesos pré-treinados
    baseline_model = timm.create_model(
        CONFIG['ARCHITECTURE'], 
        pretrained=False, # <-- Importante: Treinando do zero
        num_classes=num_classes
    )
    baseline_model = baseline_model.to(device)
    
    # Otimizador padrão para ResNet
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    # Treinamento
    baseline_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            if isinstance(images, list): # If MultiCropTransform returns multiple views, take the first one
                images = images[0]
            elif images.dim() == 5: # Handles the case where CustomDatasetFromCSV returns a stacked tensor for single view
                images = images[:, 0]
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Avaliação
    baseline_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            if isinstance(images, list): # If MultiCropTransform returns multiple views, take the first one
                images = images[0]
            elif images.dim() == 5: # Handles the case where CustomDatasetFromCSV returns a stacked tensor for single view
                images = images[:, 0]
            images = images.to(device)
            
            outputs = baseline_model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    from sklearn.metrics import f1_score
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    # Ensure all possible classes are covered for f1_per_class
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
    
    log_message(log_filepath, f"Baseline - F1-Micro: {f1_micro:.4f}")
    return {'f1_micro': f1_micro, 'f1_per_class': f1_per_class}

# --- FUNÇÃO PRINCIPAL DO EXPERIMENTO ---

def main():
    # Setup
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    LOG_FILEPATH = os.path.join(CONFIG['OUTPUT_DIR'], 'experiment.log')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Criar pasta para salvar os pesos do pré-treinamento
    PRETRAINED_DIR = os.path.join(CONFIG['OUTPUT_DIR'], 'pretrained_weights')
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    
    log_message(LOG_FILEPATH, f"Iniciando novo experimento DINO-{CONFIG['ARCHITECTURE']} com Cross-Validation.")
    log_message(LOG_FILEPATH, f"Configurações: {CONFIG}")
    
    # Discover all unique classes (excluding the ones to ignore) from the CSV
    all_dataset_classes = set()
    with open(CONFIG['CSV_PATH'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['class_name'] not in CONFIG['CLASSES_TO_EXCLUDE']:
                all_dataset_classes.add(row['class_name'])
    
    class_names = sorted(list(all_dataset_classes))
    num_classes = len(class_names)
    
    log_message(LOG_FILEPATH, f"Classes utilizadas em todo o experimento: {class_names}")
    log_message(LOG_FILEPATH, f"Número de classes: {num_classes}")

    dino_results = []
    baseline_results = []
    best_f1_micro = -1.0
    
    # Iterate through folds based on the NUM_FOLDS config
    for fold in range(0, CONFIG['NUM_FOLDS']):
        log_message(LOG_FILEPATH, f"\n{'='*20} FOLD {fold}/{CONFIG['NUM_FOLDS']} {'='*20}")
        
        # Pre-training dataset
        pretrain_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'],
            fold=fold,
            split='train',
            transform=MultiCropTransform(),
            classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR'] # <-- ADD THIS
        )

        # Fine-tuning train dataset
        finetune_train_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'],
            fold=fold,
            split='train',
            transform=CustomDatasetFromCSV.get_default_transform(),
            classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR'] # <-- ADD THIS
        )

        # Test dataset
        test_dataset = CustomDatasetFromCSV(
            csv_file=CONFIG['CSV_PATH'],
            fold=fold,
            split='test',
            transform=CustomDatasetFromCSV.get_default_transform(),
            classes_to_exclude=CONFIG['CLASSES_TO_EXCLUDE'],
            data_dir=CONFIG['DATA_DIR'] # <-- ADD THIS
        )
        
        class_names = pretrain_dataset.classes
        num_classes = len(class_names)
        log_message(LOG_FILEPATH, f"Classes for this fold: {class_names}")
        log_message(LOG_FILEPATH, f"Number of classes for this fold: {num_classes}")
        log_message(LOG_FILEPATH, f"Train samples for Fold {fold}: {len(finetune_train_dataset)}")
        log_message(LOG_FILEPATH, f"Test samples for Fold {fold}: {len(test_dataset)}")
        
        # Determine class weights for WeightedRandomSampler for fine-tuning
        train_labels = finetune_train_dataset.labels
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)
        class_weights = {cls_idx: 1.0 / count if count > 0 else 0 for cls_idx, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        
        # DataLoaders
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        finetune_train_loader = DataLoader(
            finetune_train_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            sampler=sampler, 
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)
        
        # --- Treinamento DINO ---
        dino_model = DINO(architecture=CONFIG['ARCHITECTURE'])
        dino_trainer = DINOTrainer(dino_model, device=DEVICE, base_lr=CONFIG['LEARNING_RATE'], log_filepath=LOG_FILEPATH)
        log_message(LOG_FILEPATH, f"Starting DINO pre-training for Fold {fold}")
        dino_trainer.train(pretrain_loader, num_epochs=CONFIG['EPOCHS_PRETRAIN'])

        # Salvar o modelo pré-treinado após a conclusão
        pretrain_save_path = os.path.join(PRETRAINED_DIR, f"dino_pretrained_fold_{fold}.pth")
        torch.save(dino_trainer.model.state_dict(), pretrain_save_path)
        log_message(LOG_FILEPATH, f"DINO pre-trained model for Fold {fold} saved to: {pretrain_save_path}")
        
        log_message(LOG_FILEPATH, f"Starting DINO fine-tuning for Fold {fold}")
        dino_trainer.fine_tune(finetune_train_loader, num_classes=num_classes, epochs=CONFIG['EPOCHS_FINETUNE'])
        log_message(LOG_FILEPATH, f"Evaluating DINO model for Fold {fold}")
        fold_results = dino_trainer.evaluate(test_loader, num_classes=num_classes)
        dino_results.append(fold_results)

        # Plot t-SNE (apenas no primeiro fold)
        if fold == 1:
            log_message(LOG_FILEPATH, f"Generating t-SNE plot for Fold {fold}")
            generate_tsne_plot(dino_trainer.model, test_loader, DEVICE, f"t-SNE (DINO-{CONFIG['ARCHITECTURE']} Após Fine-Tuning)", os.path.join(CONFIG['OUTPUT_DIR'], f"tsne_dino_{CONFIG['ARCHITECTURE'].lower()}_finetuned.png"), CONFIG['TSNE_SAMPLES'], class_names)

        # Salvar o melhor modelo com base no F1-Micro
        if fold_results['f1_micro'] > best_f1_micro:
            best_f1_micro = fold_results['f1_micro']
            torch.save(dino_trainer.model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], f"best_dino_{CONFIG['ARCHITECTURE'].lower()}_model.pth"))
            log_message(LOG_FILEPATH, f"Novo melhor modelo salvo com F1-Micro: {best_f1_micro:.4f}")

        # --- Treinamento Baseline ---
        log_message(LOG_FILEPATH, f"Starting Baseline training for Fold {fold}")
        # --- ALTERADO: Chamada da função de baseline genérica ---
        baseline_fold_results = train_baseline_model(finetune_train_loader, test_loader, num_classes, CONFIG['EPOCHS_FINETUNE'], DEVICE, LOG_FILEPATH)
        baseline_results.append(baseline_fold_results)
        
    # 4. Análise final dos resultados
    log_message(LOG_FILEPATH, f"\n\n{'='*20} RESULTADOS FINAIS {'='*20}")
    
    # Métricas DINO
    dino_f1_micros = [r['f1_micro'] for r in dino_results]
    dino_f1_per_class = np.array([r['f1_per_class'] for r in dino_results])
    log_message(LOG_FILEPATH, f"\n--- Modelo DINO-{CONFIG['ARCHITECTURE']} (Pré-treinado + Fine-tuning) ---")
    log_message(LOG_FILEPATH, f"F1-Score Micro Médio: {np.mean(dino_f1_micros):.4f} ± {np.std(dino_f1_micros):.4f}")
    log_message(LOG_FILEPATH, "F1-Score Médio por Classe:")
    mean_f1_per_class_dino = np.mean(dino_f1_per_class, axis=0)
    std_f1_per_class_dino = np.std(dino_f1_per_class, axis=0)
    for i, c in enumerate(class_names):
        log_message(LOG_FILEPATH, f"   - {c}: {mean_f1_per_class_dino[i]:.4f} ± {std_f1_per_class_dino[i]:.4f}")

    # Métricas Baseline
    base_f1_micros = [r['f1_micro'] for r in baseline_results]
    base_f1_per_class = np.array([r['f1_per_class'] for r in baseline_results])
    # --- ALTERADO: Log do baseline agora é genérico ---
    log_message(LOG_FILEPATH, f"\n--- Modelo Baseline ({CONFIG['ARCHITECTURE']} from Scratch) ---")
    log_message(LOG_FILEPATH, f"F1-Score Micro Médio: {np.mean(base_f1_micros):.4f} ± {np.std(base_f1_micros):.4f}")
    log_message(LOG_FILEPATH, "F1-Score Médio por Classe:")
    mean_f1_per_class_base = np.mean(base_f1_per_class, axis=0)
    std_f1_per_class_base = np.std(base_f1_per_class, axis=0)
    for i, c in enumerate(class_names):
        log_message(LOG_FILEPATH, f"   - {c}: {mean_f1_per_class_base[i]:.4f} ± {std_f1_per_class_base[i]:.4f}")

    # 5. Teste Estatístico de Wilcoxon no F1-Micro
    log_message(LOG_FILEPATH, "\n--- Teste de Significância (Wilcoxon no F1-Micro) ---")
    try:
        stat, p_value = wilcoxon(dino_f1_micros, base_f1_micros)
        log_message(LOG_FILEPATH, f"Estatística do Teste: {stat:.4f}, P-valor: {p_value:.4f}")
        if p_value < 0.05:
            log_message(LOG_FILEPATH, "Resultado: A diferença entre os modelos é estatisticamente significante.")
        else:
            log_message(LOG_FILEPATH, "Resultado: A diferença entre os modelos NÃO é estatisticamente significante.")
    except ValueError as e:
        log_message(LOG_FILEPATH, f"Não foi possível realizar o teste de Wilcoxon: {e}")

    log_message(LOG_FILEPATH, "\nExperimento concluído.")

if __name__ == '__main__':
    main()