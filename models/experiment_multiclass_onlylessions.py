# experiment.py
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
# Adicione a importação do timm
import timm
from torch.utils.data import WeightedRandomSampler
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datasets.custom_dataset_csv import CustomDatasetFromCSV  # novo import
# Assumindo que as classes DINO e CustomDataset estão nos arquivos abaixo
from dino import DINO, DINOTrainer, MultiCropTransform, log_message
# from datasets.custom_dataset import CustomDataset

# --- CONFIGURAÇÕES DO EXPERIMENTO ATUALIZADAS ---
CONFIG = {
    'DATA_DIR': './dataset-mestrado-Gabriel', # <-- MUDE AQUI: Caminho para a pasta raiz dos seus dados
    'OUTPUT_DIR': './experiment_results_vit', # Nova pasta de saída
    'CSV_PATH': './dataset-mestrado-Gabriel/metadata.csv',  # <-- Novo: caminho para o CSV
    'ARCHITECTURE': 'resnet18', # <-- MUDOU PARA ViT
    'NUM_FOLDS': 5,
    'RANDOM_STATE': 42,
    'EPOCHS_PRETRAIN': 50, # Aumente para resultados melhores (e.g., 100+)
    'EPOCHS_FINETUNE': 50,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.0005,
    'CLASSES_TO_EXCLUDE': ['1_Normal'], # <-- Classe a ser ignorada em TODAS as fases
    'TSNE_SAMPLES': 1000,
}

# --- FUNÇÕES AUXILIARES ---

# ... (A função generate_tsne_plot permanece a mesma) ...
def generate_tsne_plot(model, dataloader, device, title, filename, num_samples, class_names):
    """Gera e salva um plot t-SNE das features do backbone do modelo."""
    log_message(os.path.join(CONFIG['OUTPUT_DIR'], 'experiment.log'), f"Gerando plot t-SNE: {title}")
    
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if images.dim() == 5:
                images = images[:, 0]
            images = images.to(device)
            features = model(images)  # Usa o backbone apenas
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if len(np.concatenate(all_labels)) >= num_samples:
                break

    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    tsne = TSNE(n_components=2, random_state=CONFIG['RANDOM_STATE'], perplexity=min(30, len(all_features)-1))
    tsne_results = tsne.fit_transform(all_features)

    # Paleta com maior contraste
    cmap = plt.get_cmap('tab20') if len(class_names) <= 10 else plt.get_cmap('gist_ncar')
    palette = plt.get_cmap("tab20") 
    colors = [palette(i % 20) for i in range(len(np.unique(all_labels)))]

    plt.figure(figsize=(12, 8))
    for class_index in range(len(class_names)):
        indices = (all_labels == class_index)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    label=class_names[class_index], alpha=0.7, s=40, color=colors[class_index % len(colors)])

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    # Legenda fora do gráfico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Classes")
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def train_baseline_vit(train_loader, test_loader, num_classes, epochs, device, log_filepath):
    """Treina e avalia um modelo ViT-B/16 do zero como baseline."""
    log_message(log_filepath, "\nIniciando treinamento do modelo Baseline (ViT-B/16 from scratch)")
    
    # Modelo baseline ViT sem pesos pré-treinados
    baseline_model = timm.create_model(
        CONFIG['ARCHITECTURE'], 
        pretrained=False, # <-- Importante: Treinando do zero
        num_classes=num_classes
    )
    baseline_model = baseline_model.to(device)
    
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4) # LR menor para ViT
    criterion = nn.CrossEntropyLoss()
    
    # Treinamento
    baseline_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            if images.dim() == 5: images = images[:, 0]
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
            if images.dim() == 5: images = images[:, 0]
            images = images.to(device)
            
            outputs = baseline_model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    from sklearn.metrics import f1_score
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
    
    log_message(log_filepath, f"Baseline - F1-Micro: {f1_micro:.4f}")
    return {'f1_micro': f1_micro, 'f1_per_class': f1_per_class}

# --- FUNÇÃO PRINCIPAL DO EXPERIMENTO ---

def main():
    # Setup
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    LOG_FILEPATH = os.path.join(CONFIG['OUTPUT_DIR'], 'experiment.log')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_message(LOG_FILEPATH, "Iniciando novo experimento DINO-ViT com Cross-Validation.")
    log_message(LOG_FILEPATH, f"Configurações: {CONFIG}")
    

    # Descobrindo as classes a partir do CSV
    with open(CONFIG['CSV_PATH'], 'r') as f:
        reader = csv.DictReader(f)
        all_classes = sorted(list(set(row['class_name'] for row in reader if row['class_name'] not in CONFIG['CLASSES_TO_EXCLUDE'])))

    class_names = all_classes
    num_classes = len(class_names)
    
    # 1. Carregar dados, JÁ EXCLUINDO a classe 'Normal'
    all_available_classes = [d for d in os.listdir(CONFIG['DATA_DIR']) if os.path.isdir(os.path.join(CONFIG['DATA_DIR'], d))]
    classes_to_use = sorted(list(set(all_available_classes) - set(CONFIG['CLASSES_TO_EXCLUDE'])))
    
    log_message(LOG_FILEPATH, f"Classes utilizadas em todo o experimento: {classes_to_use}")
    
    # CustomDataset para listar todas as imagens e labels das classes permitidas
    full_dataset_helper = CustomDataset(
        root_dir=CONFIG['DATA_DIR'], 
        split='train', 
        train_ratio=1.0, # Carrega 100% dos dados permitidos
        allowed_classes=classes_to_use
    )
    
    image_paths = full_dataset_helper.image_paths
    labels = full_dataset_helper.labels # Labels já são 0, 1, 2...
    class_names = full_dataset_helper.classes
    num_classes = len(class_names)
    
    log_message(LOG_FILEPATH, f"Total de imagens encontradas: {len(image_paths)}")
    log_message(LOG_FILEPATH, f"Número de classes: {num_classes}")

    # 2. Iniciar o loop de Cross-Validation
    skf = StratifiedKFold(n_splits=CONFIG['NUM_FOLDS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    
    dino_results = []
    baseline_results = []
    best_f1_micro = -1.0
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        log_message(LOG_FILEPATH, f"\n{'='*20} FOLD {fold+1}/{CONFIG['NUM_FOLDS']} {'='*20}")
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        print(f"\nFold {fold+1}")
        print(f"Treino: {Counter(train_labels)}")
        print(f"Teste:  {Counter(test_labels)}")
        
        # Datasets para este fold
        # Pré-treino: usa o split de treino do fold com MultiCrop
        pretrain_dataset = CustomDataset(root_dir=CONFIG['DATA_DIR'], transform=MultiCropTransform(), allowed_classes=classes_to_use)
        pretrain_dataset.image_paths = [image_paths[i] for i in train_idx]
        pretrain_dataset.labels = [labels[i] for i in train_idx]
        
        # Fine-tuning: usa o split de treino do fold com transformações simples
        finetune_train_dataset = CustomDataset(root_dir=CONFIG['DATA_DIR'], transform=CustomDataset.get_default_transform(), allowed_classes=classes_to_use)
        finetune_train_dataset.image_paths = [image_paths[i] for i in train_idx]
        finetune_train_dataset.labels = [labels[i] for i in train_idx]

        # Teste: usa o split de teste do fold
        test_dataset = CustomDataset(root_dir=CONFIG['DATA_DIR'], transform=CustomDataset.get_default_transform(), allowed_classes=classes_to_use)
        test_dataset.image_paths = [image_paths[i] for i in test_idx]
        test_dataset.labels = [labels[i] for i in test_idx]
        
        train_labels = [labels[i] for i in train_idx]
        class_counts = Counter(train_labels)
        num_samples = len(train_labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        # DataLoaders
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        finetune_train_loader = DataLoader(
            finetune_train_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            sampler=sampler, 
            num_workers=4
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
        
        # --- Treinamento DINO ---
        dino_model = DINO(architecture=CONFIG['ARCHITECTURE'])
        dino_trainer = DINOTrainer(dino_model, device=DEVICE, base_lr=CONFIG['LEARNING_RATE'], log_filepath=LOG_FILEPATH)
        dino_trainer.train(pretrain_loader, num_epochs=CONFIG['EPOCHS_PRETRAIN'])
        dino_trainer.fine_tune(finetune_train_loader, num_classes=num_classes, epochs=CONFIG['EPOCHS_FINETUNE'])
        fold_results = dino_trainer.evaluate(test_loader, num_classes=num_classes)
        dino_results.append(fold_results)

        # Plot t-SNE (apenas no primeiro fold)
        if fold == 0:
            generate_tsne_plot(dino_trainer.model, test_loader, DEVICE, "t-SNE (DINO-ViT Após Fine-Tuning)", os.path.join(CONFIG['OUTPUT_DIR'], 'tsne_dino_vit_finetuned.png'), CONFIG['TSNE_SAMPLES'], class_names)

        # Salvar o melhor modelo com base no F1-Micro
        if fold_results['f1_micro'] > best_f1_micro:
            best_f1_micro = fold_results['f1_micro']
            torch.save(dino_trainer.model.state_dict(), os.path.join(CONFIG['OUTPUT_DIR'], 'best_dino_vit_model.pth'))
            log_message(LOG_FILEPATH, f"Novo melhor modelo salvo com F1-Micro: {best_f1_micro:.4f}")

        # --- Treinamento Baseline ---
        #baseline_fold_results = train_baseline_vit(finetune_train_loader, test_loader, num_classes, CONFIG['EPOCHS_FINETUNE'], DEVICE, LOG_FILEPATH)
        #baseline_results.append(baseline_fold_results)
        
    # 4. Análise final dos resultados
    log_message(LOG_FILEPATH, f"\n\n{'='*20} RESULTADOS FINAIS {'='*20}")
    
    # Métricas DINO
    dino_f1_micros = [r['f1_micro'] for r in dino_results]
    dino_f1_per_class = np.array([r['f1_per_class'] for r in dino_results])
    log_message(LOG_FILEPATH, "\n--- Modelo DINO-ViT (Pré-treinado + Fine-tuning) ---")
    log_message(LOG_FILEPATH, f"F1-Score Micro Médio: {np.mean(dino_f1_micros):.4f} ± {np.std(dino_f1_micros):.4f}")
    log_message(LOG_FILEPATH, "F1-Score Médio por Classe:")
    mean_f1_per_class_dino = np.mean(dino_f1_per_class, axis=0)
    std_f1_per_class_dino = np.std(dino_f1_per_class, axis=0)
    for i, c in enumerate(class_names):
        log_message(LOG_FILEPATH, f"  - {c}: {mean_f1_per_class_dino[i]:.4f} ± {std_f1_per_class_dino[i]:.4f}")

    # # Métricas Baseline
    # base_f1_micros = [r['f1_micro'] for r in baseline_results]
    # base_f1_per_class = np.array([r['f1_per_class'] for r in baseline_results])
    # log_message(LOG_FILEPATH, "\n--- Modelo Baseline (ViT-B/16 from Scratch) ---")
    # log_message(LOG_FILEPATH, f"F1-Score Micro Médio: {np.mean(base_f1_micros):.4f} ± {np.std(base_f1_micros):.4f}")
    # log_message(LOG_FILEPATH, "F1-Score Médio por Classe:")
    # mean_f1_per_class_base = np.mean(base_f1_per_class, axis=0)
    # std_f1_per_class_base = np.std(base_f1_per_class, axis=0)
    # for i, c in enumerate(class_names):
    #     log_message(LOG_FILEPATH, f"  - {c}: {mean_f1_per_class_base[i]:.4f} ± {std_f1_per_class_base[i]:.4f}")

    # 5. Teste Estatístico de Wilcoxon no F1-Micro
    # log_message(LOG_FILEPATH, "\n--- Teste de Significância (Wilcoxon no F1-Micro) ---")
    # try:
    #     stat, p_value = wilcoxon(dino_f1_micros, base_f1_micros)
    #     log_message(LOG_FILEPATH, f"Estatística do Teste: {stat:.4f}, P-valor: {p_value:.4f}")
    #     if p_value < 0.05:
    #         log_message(LOG_FILEPATH, "Resultado: A diferença entre os modelos é estatisticamente significante.")
    #     else:
    #         log_message(LOG_FILEPATH, "Resultado: A diferença entre os modelos NÃO é estatisticamente significante.")
    # except ValueError as e:
    #     log_message(LOG_FILEPATH, f"Não foi possível realizar o teste de Wilcoxon: {e}")

    log_message(LOG_FILEPATH, "\nExperimento concluído.")

if __name__ == '__main__':
    main()