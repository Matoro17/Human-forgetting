import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fine_tune_model import FineTuneModel
from datasets.custom_dataset import CustomDataset
from torchvision import transforms


import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def fine_tune(encoder, device, num_classes, num_epochs=10):
    # Define transformations, including conversion to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize using ImageNet stats
    ])
    
    train_set = CustomDataset(root_dir=DATASET_DIR, split='train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    
    fine_tune_model = FineTuneModel(encoder, num_classes).to(device)
    optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.0003)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        fine_tune_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = fine_tune_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')
    
    torch.save(fine_tune_model.state_dict(), 'checkpoints/fine_tune_model.pth')
    return fine_tune_model

def fine_tune_model(model, dataset):
    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify last layer for fine-tuning
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Fine-tune the model on the local dataset
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(fine_tune_epochs):
        for batch in dataset:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.targets)
            loss.backward()
            optimizer.step()

    return model
