import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.dino import DINO
from custom_dataset import CustomDataset
from utils import save_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

def train_dino(device):
    # Define data transformation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load custom dataset
    train_dataset = CustomDataset(root_dir=DATASET_DIR, transform=transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize DINO model
    dino_model = DINO().to(device)
    optimizer = optim.Adam(dino_model.student.parameters(), lr=0.0003)
    
    # Early stopping parameters
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        dino_model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            
            # Forward pass through the model
            student_output, teacher_output = dino_model(images)
            
            # Calculate the DINO loss
            loss = dino_model.loss(student_output, teacher_output)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher weights using momentum
            dino_model.update_teacher()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        
        # Early stopping logic
        if avg_loss < best_loss - EARLY_STOPPING_DELTA:
            best_loss = avg_loss
            epochs_no_improve = 0
            # save_model(dino_model, optimizer, epoch, 'training/dino_best_checkpoint.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break
    
    return dino_model.student

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dino(device)
