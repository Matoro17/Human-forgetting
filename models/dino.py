import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, ViTConfig
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
import os
import sys
from dotenv import load_dotenv

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


from datasets.custom_dataset import CustomDataset

from utils.evaluation import evaluate, save_metrics_to_csv

# Load environment variables from .env file
load_dotenv()

USE_ALL_SUBCLASSES = os.getenv("USE_ALL_SUBCLASSES", "true").lower() == "true"
SUBCLASSES = os.getenv("SUBCLASSES", "AZAN,HE,PAS").split(",") if not USE_ALL_SUBCLASSES else None
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))
USE_GLOMERULI_PRETRAINING = False

class DINO(nn.Module):
    def __init__(self, student_arch='resnet50', teacher_arch='resnet50', use_pretrained=True, out_dim=256, momentum=0.996):
        super(DINO, self).__init__()
        self.student = self._create_network(student_arch, use_pretrained, out_dim)
        self.teacher = self._create_network(teacher_arch, use_pretrained, out_dim)
        self.momentum = momentum
        self._initialize_teacher()

    def _create_network(self, arch, use_pretrained, out_dim):
        if arch == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            backbone.fc = nn.Identity()
            feature_dim = 2048  # ResNet50's final feature dimension is fixed at 2048
            projection_head = nn.Sequential(
                nn.Linear(feature_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_dim)
            )
            return nn.Sequential(backbone, projection_head)
        
        elif 'vit' in arch.lower():
            config = ViTConfig.from_pretrained(arch)
            backbone = ViTModel(config)
            feature_dim = config.hidden_size
            projection_head = nn.Sequential(
                nn.Linear(feature_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_dim)
            )
            return nn.Sequential(backbone, projection_head)
        
        else:
            raise ValueError("Unsupported architecture")

    def _initialize_teacher(self):
        """Initialize teacher network with student parameters."""
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False

    def update_teacher(self):
        """Update teacher network using momentum update."""
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.data = self.momentum * teacher_param.data + (1.0 - self.momentum) * student_param.data

    def forward(self, x):
        # Forward pass through student network
        if isinstance(self.student[0], models.resnet.ResNet):
            student_features = self.student[0](x)  # Get features from backbone
            student_output = self.student[1](student_features)  # Pass through projection head
        else:  # ViT case
            student_features = self.student[0](x).last_hidden_state[:, 0, :]  # Get CLS token
            student_output = self.student[1](student_features)

        # Forward pass through teacher network (no gradient)
        with torch.no_grad():
            if isinstance(self.teacher[0], models.resnet.ResNet):
                teacher_features = self.teacher[0](x)
                teacher_output = self.teacher[1](teacher_features)
            else:  # ViT case
                teacher_features = self.teacher[0](x).last_hidden_state[:, 0, :]
                teacher_output = self.teacher[1](teacher_features)

        return student_output, teacher_output

    def loss(self, student_output, teacher_output):
        """Compute DINO loss between student and teacher outputs."""
        student_output = F.normalize(student_output, dim=-1)
        teacher_output = F.normalize(teacher_output, dim=-1)
        loss = 2 - 2 * (student_output * teacher_output).sum(dim=-1).mean()
        return loss

class DINOTrainer:
    def __init__(self, model, train_loader, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.best_loss = float('inf')
        self.early_stopping_counter = 0

    def train(self, epochs=NUM_EPOCHS):
        # Your existing train method remains unchanged
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, _ in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x = x.to(self.device)
                student_output, teacher_output = self.model(x)
                loss = self.model.loss(student_output, teacher_output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.update_teacher()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            # Early stopping logic
            if avg_loss < self.best_loss - EARLY_STOPPING_DELTA:
                self.best_loss = avg_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def fine_tune(self, fine_tune_loader, num_classes, epochs=NUM_EPOCHS):
        # Add classification head to the model
        feature_dim = 256  # This matches the out_dim from DINO initialization
        self.model.classification_head = nn.Linear(feature_dim, num_classes).to(self.device)
        
        # Initialize new optimizer with all trainable parameters
        self.optimizer = Adam([
            {'params': self.model.student.parameters()},
            {'params': self.model.classification_head.parameters()}
        ], lr=1e-3)

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        best_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            with tqdm(fine_tune_loader, desc=f"Fine-tuning Epoch {epoch + 1}/{epochs}") as pbar:
                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Get DINO embeddings
                    student_output, _ = self.model(x)
                    
                    # Pass through classification head
                    logits = self.model.classification_head(student_output)
                    
                    # Calculate loss
                    loss = criterion(logits, y)
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update progress bar
                    total_loss += loss.item()
                    avg_loss = total_loss / (pbar.n + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            avg_epoch_loss = total_loss / len(fine_tune_loader)
            print(f"Fine-tuning Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

            # Early stopping logic
            if avg_epoch_loss < best_loss - EARLY_STOPPING_DELTA:
                best_loss = avg_epoch_loss
                early_stopping_counter = 0
                # You might want to save the best model here
                # torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                student_output, _ = self.model(x)
                logits = self.model.classification_head(student_output)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate standard deviation of F1 scores (for multi-class)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        f1_std = np.std(f1_per_class)

        # Save metrics to CSV
        metrics = {
            "f1": f1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_std": f1_std,
            "confusion_matrix": conf_matrix
        }
        save_metrics_to_csv(metrics, 'metrics.csv')

        return metrics

if __name__ == '__main__':
    # Set up dataset directory and transformations
    DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    custom_dataset = CustomDataset(
        root_dir=DATASET_DIR, 
        transform=transform, 
        subclasses=SUBCLASSES
    )
    fine_tune_loader = DataLoader(
        custom_dataset, 
        batch_size=32, 
        shuffle=True
    )
    
    test_dataset = CustomDataset(
        root_dir=DATASET_DIR, 
        split='test', 
        transform=transform, 
        subclasses=SUBCLASSES
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # Get number of classes from dataset
    num_classes = len(custom_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Train and evaluate ResNet50 model
    print("\n=== Training ResNet50 DINO model ===")
    dino_resnet = DINO(
        student_arch='resnet50',
        teacher_arch='resnet50',
        use_pretrained=True,
        out_dim=256
    )
    trainer_resnet = DINOTrainer(
        dino_resnet,
        fine_tune_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if USE_GLOMERULI_PRETRAINING:
        print("Starting pre-training with glomeruli dataset...")
        # Uncomment and implement glomeruli pretraining if needed
        # glomeruli_dataset = load_dataset("path_to_glomeruli_dataset")
        # glomeruli_loader = DataLoader(glomeruli_dataset["train"], batch_size=32, shuffle=True)
        # trainer_resnet.train(epochs=NUM_EPOCHS)

    print("Starting fine-tuning on custom dataset...")
    trainer_resnet.fine_tune(
        fine_tune_loader,
        num_classes=num_classes,
        epochs=NUM_EPOCHS
    )

    print("Evaluating ResNet50 model...")
    resnet_metrics = trainer_resnet.evaluate(test_loader)
    print("\nResNet50 Metrics:")
    for metric, value in resnet_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(resnet_metrics['confusion_matrix'])

    # Train and evaluate ViT model
    print("\n=== Training ViT DINO model ===")
    dino_vit = DINO(
        student_arch='google/vit-base-patch16-224',
        teacher_arch='google/vit-base-patch16-224',
        use_pretrained=True,
        out_dim=256
    )
    trainer_vit = DINOTrainer(
        dino_vit,
        fine_tune_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if USE_GLOMERULI_PRETRAINING:
        print("Starting pre-training with glomeruli dataset...")
        trainer_vit.train(epochs=NUM_EPOCHS)

    print("Starting fine-tuning on custom dataset...")
    trainer_vit.fine_tune(
        fine_tune_loader,
        num_classes=num_classes,
        epochs=NUM_EPOCHS
    )

    print("Evaluating ViT model...")
    vit_metrics = trainer_vit.evaluate(test_loader)
    print("\nViT Metrics:")
    for metric, value in vit_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(vit_metrics['confusion_matrix'])