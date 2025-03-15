import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    ViT_B_16_Weights, Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam, AdamW
from tqdm import tqdm
from sklearn.model_selection import KFold
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
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
log_filename = f"ViT_training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(os.getcwd(), log_filename)

# Load environment variables
load_dotenv()

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

class ViTDINO(nn.Module):
    def __init__(self, architecture='vit_b_16', use_pretrained=True, out_dim=256, momentum=0.996):
        super(ViTDINO, self).__init__()
        self.architecture = architecture
        self.student, in_features = self._create_backbone(architecture, use_pretrained)
        self.teacher, _ = self._create_backbone(architecture, use_pretrained)
        self.momentum = momentum
        
        # Add projection heads
        self.student_proj = self._create_projection_head(in_features, out_dim)
        self.teacher_proj = self._create_projection_head(in_features, out_dim)
        
        self._initialize_teacher()
        self.classification_head = None

    def _create_backbone(self, architecture, use_pretrained):
        if architecture == 'vit_b_16':
            model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 768
            model.heads = nn.Identity()
        elif architecture == 'swin_t':
            model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 768
            model.head = nn.Identity()
        elif architecture == 'swin_s':
            model = models.swin_s(weights=Swin_S_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 768
            model.head = nn.Identity()
        elif architecture == 'swin_b':
            model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 1024
            model.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        return model, in_features

    def _create_projection_head(self, in_features, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 4096),
            nn.GELU(),
            nn.Linear(4096, out_dim))
    
    def _initialize_teacher(self):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False
        for s_param, t_param in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    def update_teacher(self):
        with torch.no_grad():
            for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
                t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data
            for s_param, t_param in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
                t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data

    def forward(self, x):
        student_features = self.student(x)
        student_out = self.student_proj(student_features)
        
        with torch.no_grad():
            teacher_features = self.teacher(x)
            teacher_out = self.teacher_proj(teacher_features)
            
        return student_out, teacher_out

    def loss(self, student, teacher):
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        return 2 - 2 * (student * teacher).sum(dim=-1).mean()

class ViTTrainer:
    def __init__(self, model, device='cuda', lr=1e-4, weight_decay=0.04):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW([
            {'params': self.model.student.parameters(), 'lr': lr},
            {'params': self.model.student_proj.parameters(), 'lr': lr}
        ], weight_decay=weight_decay)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.train_time = 0.0
        self.fine_tune_time = 0.0

    def train(self, train_loader, epochs):
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                student, teacher = self.model(x)
                loss = self.model.loss(student, teacher)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.update_teacher()
                
                total_loss += loss.item()

            epoch_time = time.time() - epoch_start
            self.train_time += epoch_time
            
            avg_loss = total_loss / len(train_loader)
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

    def fine_tune(self, train_loader, num_classes, epochs):
        self.model.classification_head = nn.Linear(256, num_classes).to(self.device)
        self.optimizer = Adam([
            {'params': self.model.student.parameters(), 'lr': 1e-4},
            {'params': self.model.classification_head.parameters(), 'lr': 1e-3}
        ])
        
        class_counts = torch.bincount(torch.tensor(train_loader.dataset.dataset.labels))
        class_weights = 1. / class_counts.float()
        class_weights = class_weights.to(self.device)
                
        criterion = nn.CrossEntropyLoss(weight=class_weights)        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                student_out, _ = self.model(x)
                logits = self.model.classification_head(student_out)
                loss = criterion(logits, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            log_message(log_filepath, f"Fine-tune Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            # Early stopping
            if avg_loss < self.best_loss - EARLY_STOPPING_DELTA:
                self.best_loss = avg_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    log_message(log_filepath, f"Early stopping at epoch {epoch+1}")
                    break
        
        self.fine_tune_time = time.time() - start_time

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.cpu()
                student_out, _ = self.model(x)
                logits = self.model.classification_head(student_out)
                preds = torch.argmax(logits, dim=1).cpu()
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0,1]
        )
        metrics = {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_positive': f1[1],
            'recall_positive': recall[1],
            'precision_positive': precision[1],
            'support_positive': np.sum(y_true),
            # Keep original metrics for compatibility
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        log_message(log_filepath, f"Metrics: {metrics}")
        return metrics

def get_transform(architecture):
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if 'vit' in architecture:
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif 'swin' in architecture:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    return transforms.Compose(base_transforms + [norm])

def run_experiment(architecture, device):
    log_message(log_filepath,f"\n{'='*40}\nExperiment with Architecture: {architecture}\n{'='*40}")
    
    # Start tracking energy consumption
    tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
    tracker.start()
    start_time = time.time()

    transform = get_transform(architecture)
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
    model = ViTDINO(architecture=architecture)
    trainer = ViTTrainer(model, device=device)
    
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
    # energy_consumed = emissions.energy_consumed if emissions else 0.0
    co2_emissions = emissions
    
    metrics.update({
        'training_time': total_time,
        # 'energy_consumed': energy_consumed,
        'co2_emissions': co2_emissions,
        'detailed_training_time': trainer.train_time,
        'fine_tune_time': trainer.fine_tune_time
    })
    
    # print(metrics)
    
    log_message(log_filepath,f"\nMetrics for {architecture}:")
    log_message(log_filepath,f"Training Time: {total_time:.2f}s (Pretrain: {trainer.train_time:.2f}s, Fine-tune: {trainer.fine_tune_time:.2f}s)")
    # log_message(log_filepath,f"Energy Consumed: {metrics['energy_consumed']:.4f}kWh")
    log_message(log_filepath,f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    log_message(log_filepath,f"CO2 Emissions: {metrics['co2_emissions']:.4f}kg")
    
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    architectures = ['vit_b_16', 'swin_t', 'swin_s', 'swin_b']
    
    results = {}
    for arch in architectures:
        try:
            results[arch] = run_experiment(arch, device)
        except Exception as e:
            log_message(log_filepath,f"Failed to run experiment for {arch}: {str(e)}")
    
    save_metrics_to_txt(results, "vit_architecture_comparison.txt")
    log_message(log_filepath,"\nViT comparison saved to vit_architecture_comparison.txt")

if __name__ == "__main__":
    main()