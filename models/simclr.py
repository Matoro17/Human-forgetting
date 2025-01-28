import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from utils.evaluation import evaluate, save_metrics_to_csv

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = labels.to(z.device)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = similarity_matrix[range(2 * batch_size), labels]
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(z.device)

    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def train_ssl(model, dataloader, optimizer, epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for (x, _) in dataloader:
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0).to(device)
            z_i, z_j = model(x[:len(x)//2]), model(x[len(x)//2:])

            loss = nt_xent_loss(z_i, z_j)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

def fine_tune(model, dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model.backbone(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Fine-tune Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

def get_model(option, backbone_name='resnet50', projection_dim=128, pretrained=True, simclr_pretrained_path=None):
    if option == 'pretrained_ssl':  
        # Pretrained backbone with SimCLR for fine-tuning
        backbone = models.__dict__[backbone_name](pretrained=pretrained)
        model = SimCLR(backbone, projection_dim)

    elif option == 'retrain_ssl':  
        # Load a pretrained SimCLR model and continue SSL training
        if simclr_pretrained_path is None:
            raise ValueError("Provide a path to the pretrained SimCLR model for 'retrain_ssl'")
        
        checkpoint = torch.load(simclr_pretrained_path)
        backbone = models.__dict__[backbone_name](pretrained=False)
        model = SimCLR(backbone, projection_dim)
        model.load_state_dict(checkpoint)

    elif option == 'ssl_only':  
        # Fresh SSL training without any pre-training
        backbone = models.__dict__[backbone_name](pretrained=False)
        model = SimCLR(backbone, projection_dim)

    elif option == 'pretrained_finetune':
        # Pretrained backbone without SimCLR, directly fine-tuned
        backbone = models.__dict__[backbone_name](pretrained=pretrained)
        model = backbone

    else:
        raise ValueError("Invalid option. Choose from 'pretrained_ssl', 'retrain_ssl', 'ssl_only', or 'pretrained_finetune'.")
    
    return model

from datasets.custom_dataset import CustomDataset

def main():
    DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(root_dir=DATASET_DIR, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Dataset split for training and testing
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Option 1: Pretrained + SSL + Fine-tune
    print("Starting Pretrained SimCLR + SSL + Fine-tune")
    model = get_model('pretrained_ssl')
    optimizer = Adam(model.parameters(), lr=1e-3)
    train_ssl(model, train_loader, optimizer, epochs=10)
    fine_tune(model, train_loader, nn.CrossEntropyLoss(), optimizer, epochs=10)
    
    print("Evaluating Pretrained SimCLR + SSL + Fine-tune")
    evaluate_and_save(model, test_loader, class_names, num_classes)

    # Option 2: Retrain existing SSL model + Fine-tune
    print("Starting SimCLR Retraining + Fine-tune")
    model = get_model('retrain_ssl', simclr_pretrained_path='path_to_simclr_checkpoint.pth')
    optimizer = Adam(model.parameters(), lr=1e-3)
    train_ssl(model, train_loader, optimizer, epochs=10)
    fine_tune(model, train_loader, nn.CrossEntropyLoss(), optimizer, epochs=10)
    
    print("Evaluating SimCLR Retraining + Fine-tune")
    evaluate_and_save(model, test_loader, class_names, num_classes)

    # Option 3: SSL from scratch + Fine-tune
    print("Starting SSL from Scratch + Fine-tune")
    model = get_model('ssl_only')
    optimizer = Adam(model.parameters(), lr=1e-3)
    train_ssl(model, train_loader, optimizer, epochs=10)
    fine_tune(model, train_loader, nn.CrossEntropyLoss(), optimizer, epochs=10)

    print("Evaluating SSL from Scratch + Fine-tune")
    evaluate_and_save(model, test_loader, class_names, num_classes)


def evaluate_and_save(model, test_loader, class_names, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Evaluate the model and extract F1 scores
    f1_per_class, macro_f1, weighted_f1 = evaluate(model, device, test_loader, num_classes, class_names)

    print("\nF1 Scores per class:")
    for class_name, score in f1_per_class.items():
        print(f"{class_name}: {score:.4f}")
    print(f"Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    # Save the F1 scores for each class
    save_metrics_to_csv(f1_per_class, 'f1_scores_per_class.csv')


if __name__ == '__main__':
    main()