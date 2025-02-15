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
        
        # Initialize student and teacher networks
        self.student = self._create_network(student_arch, use_pretrained, out_dim)
        self.teacher = self._create_network(teacher_arch, use_pretrained, out_dim)
        
        # Initialize teacher parameters with student parameters
        self._initialize_teacher()

        # Momentum for teacher update
        self.momentum = momentum

    def _create_network(self, arch, use_pretrained, out_dim):
        if arch == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            backbone.fc = nn.Identity()
            feature_dim = self._get_backbone_output_dim(backbone)
            projection_head = nn.Sequential(
                nn.Linear(feature_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_dim)
            )
            return nn.Sequential(backbone, projection_head)
        
        elif 'vit' in arch.lower():  # Check if 'vit' is in the architecture name (case-insensitive)
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
            raise ValueError("Unsupported architecture. Choose 'resnet50' or a ViT model (e.g., 'google/vit-base-patch16-224').")

    def _get_backbone_output_dim(self, backbone):
        # Run a dummy input to determine the output feature size
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = backbone(dummy_input)
        return features.shape[1]

    def _initialize_teacher(self):
        # Copy student parameters to teacher
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False  # Freeze the teacher parameters

    def update_teacher(self):
        # Update the teacher parameters with momentum
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1.0 - self.momentum) * student_param.data

    def forward(self, x):
        # Forward pass through the student network
        student_output = self.student(x)
        
        # No gradient for the teacher
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        # Return both student and teacher outputs
        return student_output, teacher_output

    def loss(self, student_output, teacher_output):
        # Apply DINO loss (for simplicity, using cosine similarity)
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

    def fine_tune(self, fine_tune_loader, epochs=NUM_EPOCHS):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in tqdm(fine_tune_loader, desc=f"Fine-tuning Epoch {epoch + 1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                student_output, _ = self.model(x)
                loss = criterion(student_output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Fine-tuning Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(fine_tune_loader):.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                student_output, _ = self.model(x)
                preds = torch.argmax(student_output, dim=1)
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
    DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load your custom dataset
    custom_dataset = CustomDataset(root_dir=DATASET_DIR, transform=transform, subclasses=SUBCLASSES)
    fine_tune_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
    test_dataset = CustomDataset(root_dir=DATASET_DIR, split='test', transform=transform, subclasses=SUBCLASSES)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize DINO with ResNet50
    dino_resnet = DINO(student_arch='resnet50', teacher_arch='resnet50')
    trainer_resnet = DINOTrainer(dino_resnet, fine_tune_loader)  # Use fine_tune_loader as a placeholder

    # Optional: Pre-train with glomeruli dataset
    # if USE_GLOMERULI_PRETRAINING:
    #     # print("Starting pre-training with glomeruli dataset...")
    #     # glomeruli_dataset = load_dataset("path_to_glomeruli_dataset")
    #     # glomeruli_loader = DataLoader(glomeruli_dataset["train"], batch_size=32, shuffle=True)
    #     # trainer_resnet.train(epochs=NUM_EPOCHS)

    # Fine-tune on your custom dataset
    print("Starting fine-tuning on custom dataset...")
    trainer_resnet.fine_tune(fine_tune_loader, epochs=NUM_EPOCHS)

    # Evaluate on your test dataset
    print("Evaluating on test dataset...")
    resnet_metrics = trainer_resnet.evaluate(test_loader)
    print("ResNet50 Metrics:")
    print(resnet_metrics)

    # Repeat for ViT
    dino_vit = DINO(student_arch='google/vit-base-patch16-224', teacher_arch='google/vit-base-patch16-224')
    trainer_vit = DINOTrainer(dino_vit, fine_tune_loader)  # Use fine_tune_loader as a placeholder

    if USE_GLOMERULI_PRETRAINING:
        print("Starting pre-training with glomeruli dataset...")
        trainer_vit.train(epochs=NUM_EPOCHS)

    print("Starting fine-tuning on custom dataset...")
    trainer_vit.fine_tune(fine_tune_loader, epochs=NUM_EPOCHS)

    print("Evaluating on test dataset...")
    vit_metrics = trainer_vit.evaluate(test_loader)
    print("ViT Metrics:")
    print(vit_metrics)