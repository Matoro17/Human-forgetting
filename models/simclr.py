import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import numpy as np

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils.evaluation import evaluate, save_metrics_to_csv

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

USE_ALL_SUBCLASSES = os.getenv("USE_ALL_SUBCLASSES", "true").lower() == "true"
SUBCLASSES = os.getenv("SUBCLASSES", "AZAN,HE,PAS").split(",") if not USE_ALL_SUBCLASSES else None
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

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
    batch_size = z_i.shape[0]
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    mask = torch.eye(2 * batch_size, device=similarity_matrix.device).bool()
    
    positives = similarity_matrix.masked_select(mask).view(2 * batch_size, -1)
    negatives = similarity_matrix.masked_select(~mask).view(2 * batch_size, -1)

    labels = torch.zeros(2 * batch_size).long().to(z_i.device)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    loss = F.cross_entropy(logits, labels)
    return loss

def train_ssl(model, dataloader, optimizer, epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for (x, _) in dataloader:
            x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0).to(device)
            _, z_i = model(x[:len(x) // 2])
            _, z_j = model(x[len(x) // 2:])

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
    print("class_names: " + str(class_names))
    print("num_classes: " + str(num_classes))

    # Dataset split for training and testing
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # # Option 1: Pretrained + SSL + Fine-tune
    # print("Starting Pretrained SimCLR + SSL + Fine-tune")
    # model = get_model('pretrained_ssl')
    # optimizer = Adam(model.parameters(), lr=1e-3)
    # train_ssl(model, train_loader, optimizer, epochs=NUM_EPOCHS)
    
    # # Nested K-Fold Cross-Validation
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    #     print(f"Fold {fold + 1}")
    #     train_subset = Subset(train_dataset, train_idx)
    #     val_subset = Subset(train_dataset, val_idx)
        
    #     train_loader_fold = DataLoader(train_subset, batch_size=32, shuffle=True)
    #     val_loader_fold = DataLoader(val_subset, batch_size=32, shuffle=False)
        
    #     fine_tune(model, train_loader_fold, nn.CrossEntropyLoss(), optimizer, epochs=NUM_EPOCHS)
    #     evaluate_and_save(model, val_loader_fold, class_names, num_classes)
    
    # print("Evaluating Pretrained SimCLR + SSL + Fine-tune")
    # evaluate_and_save(model, test_loader, class_names, num_classes)

    # # Option 3: SSL from scratch + Fine-tune
    # print("Starting SSL from Scratch + Fine-tune")
    # model = get_model('ssl_only')
    # optimizer = Adam(model.parameters(), lr=1e-3)
    # train_ssl(model, train_loader, optimizer, epochs=NUM_EPOCHS)
    
    # # Nested K-Fold Cross-Validation
    # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    #     print(f"Fold {fold + 1}")
    #     train_subset = Subset(train_dataset, train_idx)
    #     val_subset = Subset(train_dataset, val_idx)
        
    #     train_loader_fold = DataLoader(train_subset, batch_size=32, shuffle=True)
    #     val_loader_fold = DataLoader(val_subset, batch_size=32, shuffle=False)
        
    #     fine_tune(model, train_loader_fold, nn.CrossEntropyLoss(), optimizer, epochs=NUM_EPOCHS)
    #     evaluate_and_save(model, val_loader_fold, class_names, num_classes)
    
    # print("Evaluating SSL from Scratch + Fine-tune")
    # evaluate_and_save(model, test_loader, class_names, num_classes)

    # Binary Classifier Evaluation
    print("Starting Binary Classifier Evaluation: 11_necrose_fibrinoide")
    binary_dataset = CustomDataset(root_dir=DATASET_DIR, transform=transform, binary_classification=True, positive_classes=["11_necrose_fibrinoide"])

    binary_train_dataset, binary_test_dataset = train_test_split(binary_dataset, test_size=0.2, random_state=42)
    
    binary_train_loader = DataLoader(binary_train_dataset, batch_size=32, shuffle=True)
    binary_test_loader = DataLoader(binary_test_dataset, batch_size=32, shuffle=False)
    
    model = get_model('pretrained_ssl')
    optimizer = Adam(model.parameters(), lr=1e-3)
    fine_tune(model, binary_train_loader, nn.CrossEntropyLoss(), optimizer, epochs=NUM_EPOCHS)
    evaluate_and_save(model, binary_test_loader, ["11_necrose_fibrinoide", "Not_necrose"], 2)

    # # Single Subclass Evaluation (e.g., AZAN)
    # print("Starting Single Subclass Evaluation (AZAN)")
    # azan_dataset = CustomDataset(root_dir=DATASET_DIR, transform=transform, subclasses=["AZAN"])
    # azan_train_dataset, azan_test_dataset = train_test_split(azan_dataset, test_size=0.2, random_state=42)
    
    # azan_train_loader = DataLoader(azan_train_dataset, batch_size=32, shuffle=True)
    # azan_test_loader = DataLoader(azan_test_dataset, batch_size=32, shuffle=False)
    
    # model = get_model('pretrained_ssl')
    # optimizer = Adam(model.parameters(), lr=1e-3)
    # fine_tune(model, azan_train_loader, nn.CrossEntropyLoss(), optimizer, epochs=NUM_EPOCHS)
    # evaluate_and_save(model, azan_test_loader, azan_dataset.classes, len(azan_dataset.classes))

def evaluate_and_save(model, test_loader, class_names, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Evaluating the model Evaluate and Save")
    print("class_names: " + str(class_names))
    print("num_classes: " + str(num_classes))
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