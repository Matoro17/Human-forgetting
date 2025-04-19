import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ViT_B_16_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
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
import math
import random

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from datasets.custom_dataset import CustomDataset
from utils.evaluation import save_metrics_to_txt, log_message

# Define log file
log_filename = f"DINO_training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_filepath = os.path.join(os.getcwd(), log_filename)


# Load environment variables
load_dotenv()

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 50))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

def set_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiCropTransform:
    def __init__(self, global_size=224, local_size=96, num_local=4):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        self.num_local = num_local

    def __call__(self, image):
        crops = [self.global1(image), self.global2(image)]
        crops += [self.local(image) for _ in range(self.num_local)]
        return crops

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class DINO(nn.Module):
    def __init__(self, architecture='resnet18'):
        super().__init__()
        self.backbone = models.__dict__[architecture](pretrained=False)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.projector = DINOHead(in_dim)
        self.classifier = None  # Will be added during fine-tuning

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        projected = self.projector(features)
        if return_features or self.classifier is None:
            return projected
        return self.classifier(projected)

class DINOTrainer:
    def __init__(self, model, device='cuda', base_lr=0.0005):
        set_seeds(42)
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        self.center = torch.zeros(1, 256, device=device)
        self.ema_alpha = 0.996

    def train(self, train_loader, num_epochs):
        teacher = DINO(self.model.architecture).to(self.device)
        teacher.load_state_dict(self.model.state_dict())
        teacher.requires_grad_(False)
        
        self.model.train()
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            momentum = 0.5 * (1. + math.cos(math.pi * epoch / num_epochs)) * 0.996
            
            for views, _ in train_loader:
                # Process multi-crop views
                global_views = [v.to(self.device) for v in views[:2]]
                local_views = [v.to(self.device) for v in views[2:]]
                all_views = global_views + local_views
                
                # Student forward
                student_out = [self.model(v) for v in all_views]
                
                # Teacher forward
                with torch.no_grad():
                    teacher_out = [teacher(v) for v in global_views]
                    teacher_out = torch.cat(teacher_out)
                    self.center = self.center * 0.9 + teacher_out.mean(0) * 0.1
                    teacher_out = (teacher_out - self.center) / 0.04
                
                # Compute loss
                loss = 0
                teacher_probs = F.softmax(teacher_out, dim=-1)
                for s in student_out:
                    student_probs = F.log_softmax(s / 0.1, dim=-1)
                    loss += -torch.mean(torch.sum(teacher_probs * student_probs, dim=-1))
                loss /= len(student_out)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # EMA update
                with torch.no_grad():
                    for t_param, s_param in zip(teacher.parameters(), self.model.parameters()):
                        t_param.data = t_param.data * momentum + s_param.data * (1 - momentum)
                
                epoch_loss += loss.item()
            
            loss_history.append(epoch_loss/len(train_loader))
        return loss_history

    def fine_tune(self, train_loader, num_classes, epochs):
        # Freeze backbone and add classifier
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Linear(256, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        return total_loss/len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                preds = torch.argmax(logits, dim=1).cpu()
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return metrics

def get_transform(architecture):
    return MultiCropTransform() if 'dino' in architecture else transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def run_experiment(architecture, device):
    log_message(log_filepath,f"\n{'='*40}\nExperiment with Architecture: {architecture}\n{'='*40}")
    
    # Start tracking energy consumption
    tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
    tracker.start()
    start_time = time.time()

    # Create dataset with architecture-specific transform
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
    model = DINO(architecture=architecture)
    trainer = DINOTrainer(model, device=device)
    
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
    architectures = ['resnet18']
    
    results = {}
    for arch in architectures:
        try:
            results[arch] = run_experiment(arch, device)
        except Exception as e:
            log_message(log_filepath,f"Failed to run experiment for {arch}: {str(e)}")
    
    # Save results
    save_metrics_to_txt(results, "architecture_comparison.csv")
    log_message(log_filepath,"\nComparison saved to architecture_comparison.csv")

def binary_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    architectures = ['resnet18']
    K_FOLDS = 2  # Number of folds for cross-validation
    
    results = {}
    
    # Load the full dataset to get all classes
    transform = get_transform(architectures[0])
    full_dataset = CustomDataset(
        root_dir=os.getenv("DATASET_DIR", "datasets/train"),
        transform=transform,
        subclasses=['PAS']
    )
    classes = full_dataset.classes
    
    for arch in architectures:
        arch_results = {}
        for class_name in classes:
            log_message(log_filepath, f"\n{'='*40}\nExperiment with Architecture: {arch}, Class: {class_name}\n{'='*40}")
            
            # Create binary dataset with current class as positive
            binary_dataset = CustomDataset(
                root_dir=os.getenv("DATASET_DIR", "datasets/train"),
                transform=transform,
                binary_classification=True,
                positive_classes=[class_name]
            )
            
            # Initialize KFold
            kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
            
            # Lists to collect metrics across folds
            fold_metrics = {
                'f1_macro': [],
                'f1_positive': [],
                'recall_positive': [],
                'precision_positive': [],
                'accuracy': [],
                'co2_emissions': [],
                'total_time': []
            }
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(binary_dataset), 1):
                log_message(log_filepath, f"\nFold {fold_idx}/{K_FOLDS}")
                
                # Start tracking energy consumption for this fold
                tracker = OfflineEmissionsTracker(country_iso_code="USA", log_level="error")
                tracker.start()
                start_time = time.time()
                
                # Create train and test subsets
                train_subset = Subset(binary_dataset, train_idx)
                test_subset = Subset(binary_dataset, test_idx)
                
                # Create dataloaders
                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
                
                # Initialize model and trainer for this fold
                model = DINO(architecture=arch)
                trainer = DINOTrainer(model, device=device)
                
                # Training phase
                trainer.train(train_loader, NUM_EPOCHS)
                trainer.fine_tune(train_loader, num_classes=2, epochs=NUM_EPOCHS)  # Binary classification
                
                # Evaluation
                metrics = trainer.evaluate(test_loader)
                
                # Stop tracking and collect emissions
                emissions = tracker.stop()
                fold_time = time.time() - start_time
                
                # Store fold metrics
                fold_metrics['f1_macro'].append(metrics['f1_macro'])
                fold_metrics['f1_positive'].append(metrics['f1_positive'])
                fold_metrics['recall_positive'].append(metrics['recall_positive'])
                fold_metrics['precision_positive'].append(metrics['precision_positive'])
                fold_metrics['accuracy'].append(metrics['accuracy'])
                fold_metrics['co2_emissions'].append(emissions)
                fold_metrics['total_time'].append(fold_time)
                
                log_message(log_filepath, 
                    f"Fold {fold_idx} Results:\n"
                    f"F1 Macro: {metrics['f1_macro']:.4f}\n"
                    f"F1 Positive: {metrics['f1_positive']:.4f}\n"
                    f"Recall Positive: {metrics['recall_positive']:.4f}\n"
                    f"Precision Positive: {metrics['precision_positive']:.4f}\n"
                    f"Accuracy: {metrics['accuracy']:.4f}\n"
                    f"Time: {fold_time:.2f}s\n"
                    f"CO2: {emissions:.4f}kg"
                )
            
            # Calculate average metrics across folds
            avg_metrics = {
                'f1_macro': np.mean(fold_metrics['f1_macro']),
                'f1_positive': np.mean(fold_metrics['f1_positive']),
                'recall_positive': np.mean(fold_metrics['recall_positive']),
                'precision_positive': np.mean(fold_metrics['precision_positive']),
                'accuracy': np.mean(fold_metrics['accuracy']),
                'co2_emissions_total': np.sum(fold_metrics['co2_emissions']),
                'total_time': np.sum(fold_metrics['total_time']),
                'co2_emissions_avg_per_fold': np.mean(fold_metrics['co2_emissions']),
                'time_avg_per_fold': np.mean(fold_metrics['total_time'])
            }
            
            arch_results[class_name] = avg_metrics
            log_message(log_filepath, f"\nAverage Metrics for {class_name}:\n{avg_metrics}")
        
        results[arch] = arch_results
    
    # Save results
    save_metrics_to_txt(results, "binary_classification_kfold_results.csv")
    log_message(log_filepath, "\nK-fold binary classification results saved to binary_classification_kfold_results.csv")


if __name__ == "__main__":
    # Initialize with proper transforms
    transform = MultiCropTransform()
    dataset = SymlinkedDataset(root_dir="your/data/path", transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Create model and trainer
    model = DINO(backbone='resnet50')
    trainer = DINOTrainer(model)

    # Training loop
    for epoch in range(100):
        loss = trainer.train_epoch(loader, epoch, 100)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
