import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os

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
        self.architecture = architecture
        self.backbone = models.__dict__[architecture](pretrained=False)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.projector = DINOHead(in_dim)
        self.classifier = None  # Added during fine-tuning

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        projected = self.projector(features)
        if return_features or self.classifier is None:
            return projected
        return self.classifier(projected)

class DINOTrainer:
    def __init__(self, model, device='cuda', base_lr=0.0005):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        
        # Initialize teacher with same architecture
        self.teacher = DINO(model.architecture).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.teacher.requires_grad_(False)
        
        # DINO parameters
        self.center = torch.zeros(1, 256, device=device)
        self.ema_alpha = 0.996

    def _update_teacher(self, momentum):
        for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def train(self, train_loader, num_epochs):
        self.model.train()
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            momentum = 0.5 * (1. + math.cos(math.pi * epoch / num_epochs)) * self.ema_alpha
            
            for views, _ in train_loader:
                # Process multi-crop views: 2 global + N local
                global_views = [v.to(self.device) for v in views[:2]]
                local_views = [v.to(self.device) for v in views[2:]]
                all_views = global_views + local_views
                batch_size = global_views[0].size(0)

                # Student forward
                student_out = [self.model(v) for v in all_views]

                # Teacher forward (only global views)
                with torch.no_grad():
                    self._update_teacher(momentum)
                    teacher_global1 = self.teacher(global_views[0])
                    teacher_global2 = self.teacher(global_views[1])
                    
                    # Center and sharpen teacher outputs
                    self.center = self.center * 0.9 + 0.1 * (teacher_global1.mean(0) + teacher_global2.mean(0))/2
                    teacher_global1 = (teacher_global1 - self.center) / 0.04
                    teacher_global2 = (teacher_global2 - self.center) / 0.04

                # Calculate loss
                loss = 0
                for i, s_out in enumerate(student_out):
                    if i < 2:  # Global views
                        t_probs = F.softmax([teacher_global1, teacher_global2][i], dim=-1)
                    else:  # Local views use averaged teacher
                        avg_teacher = (teacher_global1 + teacher_global2) / 2
                        t_probs = F.softmax(avg_teacher, dim=-1)
                    
                    student_log_probs = F.log_softmax(s_out / 0.1, dim=-1)
                    loss += -torch.mean(torch.sum(t_probs * student_log_probs, dim=-1))
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            loss_history.append(epoch_loss / len(train_loader))
        
        return loss_history

    def fine_tune(self, train_loader, num_classes, epochs):
        # Freeze backbone and add classifier
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Linear(256, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(epochs):
            total_loss = 0
            for x, y in train_loader:
                x = x[0].to(self.device)  # Use first global view for fine-tuning
                y = y.to(self.device)
                
                logits = self.model(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return total_loss / len(train_loader)

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
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

def log_message(log_filepath, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_filepath, "a") as log_file:
        log_file.write(full_message + "\n")

def save_metrics_to_txt(metrics_dict, filename):
    with open(filename, "w") as f:
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")