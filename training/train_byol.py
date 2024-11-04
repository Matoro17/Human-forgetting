import os
import sys

# Add the project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from dotenv import load_dotenv
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from byol_pytorch import BYOL
from utils.save_model import save_model

from datasets.custom_dataset import CustomDataset

# Load environment variables
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
USE_ALL_SUBCLASSES = os.getenv("USE_ALL_SUBCLASSES", "true").lower() == "true"
SUBCLASSES = os.getenv("SUBCLASSES", "AZAN,HE,PAS").split(",") if not USE_ALL_SUBCLASSES else None
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

def train_byol(device):
    # Data transformation for BYOL
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])
    
    # Load custom dataset
    train_dataset = CustomDataset(
        root_dir=DATASET_DIR, 
        transform=transform, 
        split='train', 
        subclasses=SUBCLASSES
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize the ResNet backbone and BYOL
    base_encoder = resnet50(pretrained=True)
    learner = BYOL(net=base_encoder, image_size=224, hidden_layer='avgpool')
    
    # Move model to device
    learner.to(device)
    
    # Optimizer
    optimizer = optim.Adam(learner.parameters(), lr=0.001)
    
    # Early stopping parameters
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        learner.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            loss = learner(images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            learner.update_moving_average()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        
        # Early stopping logic
        if avg_loss < best_loss - EARLY_STOPPING_DELTA:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_model(learner, optimizer, epoch, 'training/byol_best_checkpoint.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break
    
    # Return the encoder part of BYOL
    return learner.net

