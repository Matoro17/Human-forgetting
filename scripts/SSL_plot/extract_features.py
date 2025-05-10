import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np

from datasets.custom_dataset import CustomDataset

# Configurações
DATA_DIR = "./data/custom"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (sem augmentation)
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset e DataLoader
dataset = CustomDataset(root=DATA_DIR, transform=eval_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Backbone carregado
backbone = models.resnet18(pretrained=False)
backbone.fc = torch.nn.Identity()
backbone.load_state_dict(torch.load("dino_resnet18_backbone.pth"))
backbone = backbone.to(DEVICE)
backbone.eval()

features = []
labels = []

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        feats = backbone(imgs)
        features.append(feats.cpu().numpy())
        labels.append(lbls.numpy())

features = np.vstack(features)
labels = np.concatenate(labels)

np.save("features.npy", features)
np.save("labels.npy", labels)

print("Features e labels salvos como features.npy e labels.npy")
