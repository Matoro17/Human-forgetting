import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simclr import ProjectionHead
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from utils.augmentations import SimCLRDataset
from utils.losses import contrastive_loss

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
USE_ALL_SUBCLASSES = os.getenv("USE_ALL_SUBCLASSES", "true").lower() == "true"
SUBCLASSES = os.getenv("SUBCLASSES", "AZAN,HE,PAS").split(",") if not USE_ALL_SUBCLASSES else None
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

def train_simclr(device):
    encoder = Encoder().to(device)
    simclr_model = ProjectionHead(encoder).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.0003)
    
    train_set = CustomDataset(root_dir=DATASET_DIR, split='train')
    simclr_train_set = SimCLRDataset(train_set)  # Wrap with SimCLRDataset to apply augmentations
    simclr_train_loader = DataLoader(simclr_train_set, batch_size=256, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        simclr_model.train()
        running_loss = 0.0
        for x_i, x_j in simclr_train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            z_i = simclr_model(x_i)
            z_j = simclr_model(x_j)
            loss = contrastive_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(simclr_train_loader)}')

    torch.save(simclr_model.state_dict(), 'checkpoints/simclr/simclr_model.pth')
    return encoder
