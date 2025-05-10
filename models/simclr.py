import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
import os
import sys
import time
from codecarbon import OfflineEmissionsTracker
from dotenv import load_dotenv
from datetime import datetime

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from datasets.custom_dataset import CustomDataset
from utils.evaluation import save_metrics_to_txt, log_message

# Define log file
log_filename = f"SimCLR_training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(os.getcwd(), log_filename)

# Load environment variables
load_dotenv()

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

class SimCLR(nn.Module):
    def __init__(self, backbone_name='resnet18', use_pretrained=True, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = self._create_backbone(backbone_name, use_pretrained)
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.backbone.fc = nn.Identity()
        self.classification_head = None

    def _create_backbone(self, backbone_name, use_pretrained):
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.feature_dim = 512  # Output feature dimension for resnet18
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.feature_dim = 2048  # Output feature dimension for resnet50
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if use_pretrained else None)
            self.feature_dim = 2048  # Output feature dimension for resnet101
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        return backbone

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)

    def nt_xent_loss(self, z_i, z_j, temperature=0.5):
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

class SimCLRTrainer:
    def __init__(self, model, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.train_time = 0.0
        self.fine_tune_time = 0.0

    def train(self, train_loader, epochs):
        self.loss_history = []
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0).to(self.device)
                _, z_i = self.model(x[:len(x) // 2])
                _, z_j = self.model(x[len(x) // 2:])
                loss = self.model.nt_xent_loss(z_i, z_j)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            epoch_time = time.time() - epoch_start
            self.train_time += epoch_time
            
            avg_loss = total_loss / len(train_loader)
            self.loss_history.append(avg_loss)
            log_message(log_filepath, f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Early stopping
            if avg_loss < self.best_loss - EARLY_STOPPING_DELTA:
                self.best_loss = avg_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    log_message(log_filepath, f"Early stopping at epoch {epoch+1}")
                    break
        return self.loss_history

    def fine_tune(self, train_loader, num_classes, epochs):
        self.acc_history = []
        # Ensure the classification head matches the backbone's output dimension
        self.model.classification_head = nn.Linear(self.model.feature_dim, num_classes).to(self.device)
        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': 1e-4},
            {'params': self.model.classification_head.parameters(), 'lr': 1e-3}
        ])
        
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                h, _ = self.model(x)
                logits = self.model.classification_head(h)
                loss = criterion(logits, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.acc_history.append(avg_loss)
            log_message(log_filepath, f"Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        self.fine_tune_time = time.time() - start_time
        return self.acc_history

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.cpu()
                h, _ = self.model(x)
                logits = self.model.classification_head(h)
                preds = torch.argmax(logits, dim=1).cpu()
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_positive = f1_score(y_true, y_pred, average='binary')  # Positive class F1
        cm = confusion_matrix(y_true, y_pred)
        
        log_message(self.log_filepath, f"F1 Macro: {f1_macro:.4f}")
        log_message(self.log_filepath, f"F1 Positive: {f1_positive:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_positive': f1_positive,
            'confusion_matrix': cm
        }

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def run_experiment(backbone_name, device):
    log_message(log_filepath,f"\n{'='*40}\nExperiment with Backbone: {backbone_name}\n{'='*40}")
    
    # Start tracking energy consumption
    tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
    tracker.start()
    start_time = time.time()

    # Create dataset
    transform = get_transform()
    full_dataset = CustomDataset(
        root_dir=os.getenv("DATASET_DIR", "datasets/train"),
        transform=transform
    )
    labels = [y for _, y in full_dataset]
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(np.unique(labels))

    # Initialize model and trainer
    model = SimCLR(backbone_name=backbone_name)
    trainer = SimCLRTrainer(model, device=device)
    
    # Training phase
    trainer.train(train_loader, NUM_EPOCHS)
    trainer.fine_tune(train_loader, num_classes, NUM_EPOCHS)
    
    # Evaluation
    metrics = trainer.evaluate(test_loader)
    
    # Stop tracking and calculate metrics
    try:
        emissions = tracker.stop()
    except Exception as e:
        log_message(log_filepath,f"Error stopping tracker: {str(e)}")
        emissions = None
    
    total_time = time.time() - start_time
    
    # Handle missing emissions data
    co2_emissions = emissions
    
    metrics.update({
        'training_time': total_time,
        'co2_emissions': co2_emissions,
        'detailed_training_time': trainer.train_time,
        'fine_tune_time': trainer.fine_tune_time
    })
    
    log_message(log_filepath,f"\nMetrics for {backbone_name}:")
    log_message(log_filepath,f"Training Time: {total_time:.2f}s (Pretrain: {trainer.train_time:.2f}s, Fine-tune: {trainer.fine_tune_time:.2f}s)")
    log_message(log_filepath,f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    log_message(log_filepath,f"CO2 Emissions: {metrics['co2_emissions']:.4f}kg")
    
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbones = ['resnet18']
    
    results = {}
    for backbone in backbones:
        try:
            results[backbone] = run_experiment(backbone, device)
        except Exception as e:
            log_message(log_filepath,f"Failed to run experiment for {backbone}: {str(e)}")
    
    # Save results
    save_metrics_to_txt(results, "backbone_comparison.csv")
    log_message(log_filepath,"\nComparison saved to backbone_comparison.csv")

if __name__ == "__main__":
    main()