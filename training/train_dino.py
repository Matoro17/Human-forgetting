import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dino import DINO
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from utils.augmentations import SimCLRTransform  # Reusing the SimCLRTransform for data augmentation
from utils.losses import dino_loss

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def train_dino(device, num_epochs=10):
    encoder = Encoder().to(device)
    dino_model = DINO(encoder).to(device)
    optimizer = optim.Adam(dino_model.parameters(), lr=0.0003)
    
    train_set = CustomDataset(root_dir=DATASET_DIR, split='train')
    dino_train_set = SimCLRTransform()(train_set)
    dino_train_loader = DataLoader(dino_train_set, batch_size=256, shuffle=True)

    for epoch in range(num_epochs):
        dino_model.train()
        running_loss = 0.0
        for x_i, x_j in dino_train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            student_proj1, student_proj2, teacher_proj1, teacher_proj2 = dino_model(x_i, x_j)
            loss = dino_loss(student_proj1, student_proj2, teacher_proj1, teacher_proj2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dino_train_loader)}')

    torch.save(dino_model.state_dict(), 'checkpoints/dino/dino_model.pth')
    return encoder  # Return the trained encoder for fine-tuning
