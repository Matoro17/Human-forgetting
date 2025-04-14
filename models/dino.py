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

class DINO(nn.Module):
    def __init__(self, architecture='resnet50', use_pretrained=True, out_dim=256, momentum=0.996):
        super(DINO, self).__init__()
        self.architecture = architecture
        self.student = self._create_network(architecture, use_pretrained, out_dim)
        self.teacher = self._create_network(architecture, use_pretrained, out_dim)
        self.momentum = momentum
        self._initialize_teacher()
        self.classification_head = None

    def _create_network(self, architecture, use_pretrained, out_dim):
        # Create backbone
        if architecture == 'resnet18':
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 512
            backbone.fc = nn.Identity()
        elif architecture == 'resnet50':
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 2048
            backbone.fc = nn.Identity()
        elif architecture == 'resnet101':
            backbone = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 2048
            backbone.fc = nn.Identity()
        elif architecture == 'vit_b_16':
            backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None)
            in_features = 768
            backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Create projection head
        projection_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )
        
        # if self.classification_head is not None:
        #     # Initialize bias to -log((1-p)/p) where p is positive class ratio
        #     pos_ratio = len(np.where(np.array(self.labels) == 1)[0])/len(self.labels)
        #     bias_init = -torch.log(torch.tensor((1-pos_ratio)/pos_ratio))
        #     self.classification_head.bias.data = torch.tensor([-bias_init, bias_init])

        return nn.Sequential(backbone, projection_head)

    def _initialize_teacher(self):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.copy_(s_param.data)
            t_param.requires_grad = False

    def update_teacher(self):
        with torch.no_grad():
            for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
                t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data

    def forward(self, x):
        student_out = self.student(x)
        with torch.no_grad():
            teacher_out = self.teacher(x)
        return student_out, teacher_out

    def loss(self, student, teacher):
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        return 2 - 2 * (student * teacher).sum(dim=-1).mean()

class DINOTrainer:
    def __init__(self, model, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.train_time = 0.0
        self.fine_tune_time = 0.0

    def train(self, train_loader, epochs):
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.loss_history = []
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
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.acc_history = []
        self.model.classification_head = nn.Linear(256, num_classes).to(self.device)
        self.optimizer = Adam([
            {'params': self.model.student.parameters(), 'lr': 1e-4},
            {'params': self.model.classification_head.parameters(), 'lr': 1e-3}
        ])
        
        labels = [y for _, y in train_loader.dataset]  # Collect labels from all samples
        class_counts = torch.bincount(torch.tensor(labels))
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
            self.acc_history.append(avg_loss)
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
        return self.acc_history

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
            'f1_macro': f1_score(y_true, y_pred, average='binary'),
            'f1_positive': f1[1],
            'recall_positive': recall[1],
            'precision_positive': precision[1],
            'support_positive': np.sum(y_true),
            # Keep original metrics for compatibility
            'f1': f1_score(y_true, y_pred, average='binary'),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        log_message(log_filepath, f"Metrics: {metrics}")
        return metrics

def get_transform(architecture):
    if 'vit' in architecture:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
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
    binary_main()
