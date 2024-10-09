import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.byol import BYOL
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from utils.augmentations import SimCLRTransform  # Reusing the SimCLRTransform for data augmentation
from utils.losses import byol_loss

def train_byol(device, num_epochs=10):
    encoder = Encoder().to(device)
    byol_model = BYOL(encoder).to(device)
    optimizer = optim.Adam(byol_model.parameters(), lr=0.0003)
    
    train_set = CustomDataset(root_dir='../datasetMestradoGledson+gabriel', split='train')
    byol_train_set = SimCLRTransform()(train_set)
    byol_train_loader = DataLoader(byol_train_set, batch_size=256, shuffle=True)

    for epoch in range(num_epochs):
        byol_model.train()
        running_loss = 0.0
        for x_i, x_j in byol_train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            online_proj1, online_proj2, target_proj1, target_proj2 = byol_model(x_i, x_j)
            loss = byol_loss(online_proj1, online_proj2, target_proj1, target_proj2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(byol_train_loader)}')

    torch.save(byol_model.state_dict(), 'checkpoints/byol/byol_model.pth')
    return encoder  # Return the trained encoder for fine-tuning
