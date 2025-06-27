import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from models.dino import DINO, DINOTrainer, MultiCropTransform, log_message, save_metrics_to_txt
from datasets.custom_dataset import CustomDataset

# Configurações do experimento
TARGET_CLASS = "11_necrose_fibrinoide"
NEGATIVE_CLASS = "1_Normal"
ALLOWED_CLASSES = [TARGET_CLASS, NEGATIVE_CLASS]
SUBCLASSES = None
BATCH_SIZE = 16
NUM_WORKERS = 4
SSL_EPOCHS = 50
FT_EPOCHS = 50
BASE_LR = 0.0005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_FILE = f"dino_binary_{TARGET_CLASS.lower()}_vs_normal.log"
ROOT_DIR = "dataset-mestrado-Gabriel"

# Transforms
train_transform = MultiCropTransform()
eval_transform = CustomDataset.get_default_transform()

# ======================
# 1. Pré-treinamento SSL
# ======================
# Aqui usamos TODO o conjunto de dados, sem binarização
ssl_dataset = CustomDataset(
    root_dir=ROOT_DIR,
    transform=train_transform,
    split='train',
    subclasses=SUBCLASSES,
    binary_classification=False,       # importante: não binariza
    allowed_classes=None               # usa todas as classes
)
train_loader_ssl = DataLoader(ssl_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Instancia modelo e trainer
model = DINO(architecture='resnet18')
trainer = DINOTrainer(model=model, device=DEVICE)
trainer.train(train_loader_ssl, num_epochs=SSL_EPOCHS)

# ======================
# 2. Fine-tuning supervisionado
# ======================
# Apenas Normal vs TARGET_CLASS
train_dataset_ft = CustomDataset(
    root_dir=ROOT_DIR,
    transform=eval_transform,
    split='train',
    subclasses=SUBCLASSES,
    binary_classification=True,
    positive_classes=[TARGET_CLASS],
    allowed_classes=ALLOWED_CLASSES
)
train_loader_ft = DataLoader(train_dataset_ft, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Teste supervisionado com mesmas classes
test_dataset = CustomDataset(
    root_dir=ROOT_DIR,
    transform=eval_transform,
    split='test',
    subclasses=SUBCLASSES,
    binary_classification=True,
    positive_classes=[TARGET_CLASS],
    allowed_classes=ALLOWED_CLASSES
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Fine-tuning supervisionado
trainer.fine_tune(train_loader_ft, num_classes=2, epochs=FT_EPOCHS)

# Avaliação
metrics = trainer.evaluate(test_loader)

# Salvando métricas
os.makedirs("results", exist_ok=True)
result_path = f"results/binary_metrics_{TARGET_CLASS.lower()}_vs_normal.txt"
save_metrics_to_txt(
    metrics_dict={
        "resnet18": {
            TARGET_CLASS: {
                "avg_accuracy": metrics['accuracy'],
                "std_accuracy": 0.0,
                "avg_f1_macro": metrics['f1_macro'],
                "std_f1_macro": 0.0,
                "avg_f1_positive": metrics['f1_positive'],
                "std_f1_positive": 0.0,
            }
        }
    },
    filename=result_path,
    log_filepath=LOG_FILE
)
