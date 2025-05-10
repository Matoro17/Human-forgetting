import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from models.dino import DINO, DINOTrainer
from datasets.custom_dataset import CustomDataset
import numpy as np

# Configurações
DATA_DIR = "./dataset-mestrado-Gabriel"  # ajuste se necessário
BATCH_SIZE = 32
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (usar o mesmo MultiCropTransform do DINO, se disponível)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset e DataLoader
train_dataset = CustomDataset(root=DATA_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Modelo backbone ResNet18
backbone = models.resnet18(pretrained=False)
backbone.fc = torch.nn.Identity()  # remove camada final

# Instancia DINO
dino_model = DINO(backbone=backbone).to(DEVICE)
trainer = DINOTrainer(dino_model, device=DEVICE)

# Pré-treinamento
for epoch in range(NUM_EPOCHS):
    trainer.train_one_epoch(train_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] completed.")

# Salva o backbone treinado
torch.save(backbone.state_dict(), "dino_resnet18_backbone.pth")

print("Backbone salvo como dino_resnet18_backbone.pth")
