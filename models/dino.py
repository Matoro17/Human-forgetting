import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os

EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

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
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

class DINO(nn.Module):
    def __init__(self, architecture='resnet18'):
        super().__init__()
        self.architecture = architecture
        self.backbone = models.__dict__[architecture](pretrained=True)
        self.in_dim = self.backbone.fc.in_features  # Store feature dimension
        self.backbone.fc = nn.Identity()
        
        self.projector = DINOHead(self.in_dim)
        self.classifier = None  # Added during fine-tuning

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        projected = self.projector(features)
        if return_features or self.classifier is None:
            return projected
        # Use backbone features for classifier during fine-tuning
        return self.classifier(features)

class DINOTrainer:
    def __init__(self, model, device='cuda', base_lr=0.0005, 
                 log_filepath='dino_training.log',
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 early_stopping_delta=EARLY_STOPPING_DELTA):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        self.log_filepath = log_filepath
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        
        # Initialize teacher
        self.teacher = DINO(model.architecture).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.teacher.requires_grad_(False)
        
        # DINO parameters
        self.center = torch.zeros(256, device=device)
        self.ema_alpha = 0.996

        log_message(self.log_filepath, "DINO Trainer initialized")
        log_message(self.log_filepath, f"Using device: {device}")
        log_message(self.log_filepath, f"Base learning rate: {base_lr}")
        log_message(self.log_filepath, f"EMA alpha: {self.ema_alpha}")
        log_message(self.log_filepath, f"Log file: {log_filepath}")

    def _update_teacher(self, momentum):
        with torch.no_grad():
            for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def train(self, train_loader, num_epochs):
        self.model.train()
        loss_history = []
        best_loss = float('inf')
        no_improve = 0
        
        log_message(self.log_filepath, f"\nStarting DINO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            momentum = 1 - (1 - self.ema_alpha) * (math.cos(math.pi * epoch / num_epochs) + 1) / 2
            log_message(self.log_filepath, f"\nEpoch {epoch+1}/{num_epochs} - Teacher momentum: {momentum:.4f}")
            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_epoch_loss)
            for batch_idx, (views, _) in enumerate(train_loader):
                # Process multi-crop views
                global_views = [v.to(self.device) for v in views[:2]]
                local_views = [v.to(self.device) for v in views[2:]]
                all_views = global_views + local_views

                # Student forward
                student_out = [self.model(v) for v in all_views]

                # Teacher forward with momentum update
                with torch.no_grad():
                    self._update_teacher(momentum)
                    
                    # Get both global teacher outputs
                    t_global1 = self.teacher(global_views[0])
                    t_global2 = self.teacher(global_views[1])
                    
                    # Update center (detach for safety)
                    self.center = self.center * 0.9 + 0.1 * (t_global1.mean(dim=0) + t_global2.mean(dim=0)).detach()/2
                    
                    # Apply centering and temperature to teacher outputs
                    t_global1 = (t_global1 - self.center) / 0.04  # Paper uses 0.04 temperature
                    t_global2 = (t_global2 - self.center) / 0.04

                # Calculate loss
                loss = 0
                for i, s_out in enumerate(student_out):
                    if i < 2:  # Global views
                        teacher_target = [t_global1, t_global2][i]
                    else:       # Local views
                        teacher_target = (t_global1 + t_global2) / 2  # Average teacher for locals
                    
                    # CORRECTED Loss Calculation with Averaging
                    loss += F.kl_div(
                        F.log_softmax(s_out / 0.1, dim=-1),  # Student temp 0.1
                        F.softmax(teacher_target, dim=-1),
                        reduction='batchmean'
                    )
                
                loss /= len(student_out)
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # if (batch_idx+1) % 50 == 0:
                #     log_message(self.log_filepath, 
                #               f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_epoch_loss)
            if (best_loss - avg_epoch_loss) > self.early_stopping_delta:
                best_loss = avg_epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                
            log_message(self.log_filepath, 
                      f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} | "
                      f"Best: {best_loss:.4f} | Patience: {no_improve}/{self.early_stopping_patience}")
            
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath, 
                          f"Early stopping at epoch {epoch+1}")
                break
        
        log_message(self.log_filepath, "\nDINO training completed")
        return loss_history
    
    def fine_tune(self, train_loader, num_classes, epochs):
        best_loss = float('inf')
        no_improve = 0
        log_message(self.log_filepath, "\nStarting fine-tuning phase")
        
        # Freeze backbone and initialize classifier with correct input dimension
        for param in self.model.parameters():
            param.requires_grad = False
        # Use backbone's feature dimension for classifier input
        self.model.classifier = nn.Linear(self.model.in_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x[0].to(self.device)  # Use first global view
                y = y.to(self.device)
                
                logits = self.model(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx+1) % 20 == 0:
                    log_message(self.log_filepath,
                              f"Fine-tuning Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            # Early stopping check
            if (best_loss - avg_loss) > self.early_stopping_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            log_message(self.log_filepath,
                       f"Fine-tuning Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | "
                       f"Best: {best_loss:.4f} | Patience: {no_improve}/{self.early_stopping_patience}")
            
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath,
                          f"Early stopping at epoch {epoch+1}")
                break
        
        log_message(self.log_filepath, "\nFine-tuning completed")
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        
        log_message(self.log_filepath, "\nStarting evaluation on test set")
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x = x[0].to(self.device) 
                logits = self.model(x)
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

def log_message(log_filepath, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_filepath, "a") as log_file:
        log_file.write(full_message + "\n")

def save_metrics_to_txt(metrics_dict, filename, log_filepath):
    with open(filename, "w") as f:
        for arch, arch_results in metrics_dict.items():
            f.write(f"{arch}:\n")
            for class_name, metrics in arch_results.items():
                f.write(f"  {class_name}:\n")
                f.write(f"    Accuracy: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}\n")
                f.write(f"    F1 Macro: {metrics['avg_f1_macro']:.4f} ± {metrics['std_f1_macro']:.4f}\n")
                f.write(f"    F1 Positive: {metrics['avg_f1_positive']:.4f} ± {metrics['std_f1_positive']:.4f}\n")
    log_message(log_filepath, f"Metrics saved to {filename}")