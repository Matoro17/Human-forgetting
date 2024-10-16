import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simclr import ProjectionHead
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from utils.augmentations import SimCLRDataset
from utils.losses import contrastive_loss

def train_simclr(device, num_epochs=10):
    encoder = Encoder().to(device)
    simclr_model = ProjectionHead(encoder).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.0003)
    
    train_set = CustomDataset(root_dir='./datasetMestradoGledson+gabriel', split='train')
    simclr_train_set = SimCLRDataset(train_set)  # Wrap with SimCLRDataset to apply augmentations
    simclr_train_loader = DataLoader(simclr_train_set, batch_size=256, shuffle=True)

    for epoch in range(num_epochs):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(simclr_train_loader)}')

    torch.save(simclr_model.state_dict(), 'checkpoints/simclr/simclr_model.pth')
    return encoder
