#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights # Specific weights
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os
import timm # <<< Added import for timm
import copy # <<< Added missing import
from tqdm import tqdm # <<< Added missing import

# Constants from the original file (can be overridden by trainer args if needed)
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

# --- Utility Functions (from original file) ---
def log_message(log_filepath, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    # Ensure directory exists before writing
    log_dir = os.path.dirname(log_filepath)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_filepath, "a") as log_file:
        log_file.write(full_message + "\n")

def save_metrics_to_txt(metrics_dict, filename, log_filepath):
    # Ensure directory exists before writing
    file_dir = os.path.dirname(filename)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(filename, "w") as f:
        for arch, arch_results in metrics_dict.items():
            f.write(f"{arch}:\n")
            for class_name, metrics in arch_results.items():
                f.write(f"  {class_name}:\n")
                # Check if metrics exist before accessing
                acc_mean = metrics.get("avg_accuracy", float("nan"))
                acc_std = metrics.get("std_accuracy", float("nan"))
                f1_macro_mean = metrics.get("avg_f1_macro", float("nan"))
                f1_macro_std = metrics.get("std_f1_macro", float("nan"))
                f1_pos_mean = metrics.get("avg_f1_positive", float("nan"))
                f1_pos_std = metrics.get("std_f1_positive", float("nan"))
                f.write(f"    Accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n")
                f.write(f"    F1 Macro: {f1_macro_mean:.4f} ± {f1_macro_std:.4f}\n")
                f.write(f"    F1 Positive: {f1_pos_mean:.4f} ± {f1_pos_std:.4f}\n")
    log_message(log_filepath, f"Metrics saved to {filename}")

# --- Transforms (Adapted from original file + DINO standards) ---
class MultiCropTransform:
    def __init__(self, global_size=224, local_size=96, num_local=6, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4)):
        # Using ImageNet normalization for consistency, adjust if needed
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * global_size) // 2 * 2 + 1)], p=1.0),
            transforms.ToTensor(),
            normalize
        ])

        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * global_size) // 2 * 2 + 1)], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            normalize
        ])

        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * local_size) // 2 * 2 + 1)], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        self.num_local = num_local

    def __call__(self, image):
        crops = [self.global1(image), self.global2(image)]
        crops += [self.local(image) for _ in range(self.num_local)]
        return crops

# --- DINO Head (Standard DINO implementation) ---
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# --- DINO Model (Updated, Cleaned Quotes) ---
class DINO(nn.Module):
    def __init__(self, architecture='resnet18', use_pretrained=True, out_dim=65536):
        super().__init__()
        self.architecture = architecture
        self.backbone, self.feature_dim = self._create_backbone(architecture, use_pretrained)
        self.projector = DINOHead(self.feature_dim, out_dim=out_dim)
        self.classifier = None

    def _create_backbone(self, architecture, use_pretrained):
        feature_dim = 0
        print(f"Creating backbone: {architecture}, Pretrained: {use_pretrained}")
        if architecture.startswith("resnet"):
            weights = None
            if use_pretrained:
                if architecture == "resnet18": weights = ResNet18_Weights.IMAGENET1K_V1
                elif architecture == "resnet50": weights = ResNet50_Weights.IMAGENET1K_V1
                elif architecture == "resnet101": weights = ResNet101_Weights.IMAGENET1K_V1
                else: raise ValueError(f"Unsupported pretrained ResNet variant: {architecture}")
            if weights:
                backbone = models.__dict__[architecture](weights=weights)
            else:
                backbone = models.__dict__[architecture]()
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            print(f"Created ResNet backbone {architecture}. Feature dim: {feature_dim}")
        elif architecture.startswith("vit_") or architecture.startswith("swin_"):
            try:
                backbone = timm.create_model(architecture, pretrained=use_pretrained)
                feature_dim = backbone.num_features
                if hasattr(backbone, 'head'): backbone.head = nn.Identity()
                elif hasattr(backbone, 'fc'): backbone.fc = nn.Identity()
                elif hasattr(backbone, 'classifier'): backbone.classifier = nn.Identity()
                else: print(f"Warning: Could not automatically remove classifier head from timm model {architecture}.")
                print(f"Created timm backbone {architecture}. Feature dim: {feature_dim}")
            except Exception as e:
                raise ValueError(f"Error loading backbone {architecture} with timm: {e}")
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        if feature_dim == 0:
             raise ValueError(f"Feature dimension could not be determined for {architecture}")
        return backbone, feature_dim

    def forward(self, x, return_features=False, pretrain_mode=False):
        features = self.backbone(x)
        if pretrain_mode:
            return self.projector(features)
        else:
            if return_features:
                return features
            elif self.classifier is not None:
                return self.classifier(features)
            else:
                raise RuntimeError("Classifier not initialized. Call add_classification_head first for evaluation.")

    def add_classification_head(self, num_classes):
        print(f"Adding classification head with {num_classes} output classes. Input features: {self.feature_dim}")
        self.classifier = nn.Linear(self.feature_dim, num_classes)

# --- DINO Trainer (Adapted, Cleaned Quotes) ---
class DINOTrainer:
    def __init__(self, model: DINO, device='cuda', base_lr=0.0005,
                 log_filepath='dino_training.log',
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 early_stopping_delta=EARLY_STOPPING_DELTA,
                 teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        self.student_model = model.to(device)
        self.device = device
        self.base_lr = base_lr
        self.log_filepath = log_filepath
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.teacher_model = DINO(architecture=model.architecture, use_pretrained=False, out_dim=model.projector.last_layer.out_features).to(device)
        self.teacher_model.load_state_dict(self.student_model.state_dict())
        self.teacher_model.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=base_lr)
        self.center = torch.zeros(model.projector.last_layer.out_features, device=device)
        self.ema_alpha = 0.996

        log_message(self.log_filepath, "DINO Trainer initialized")
        log_message(self.log_filepath, f"Using device: {device}")
        log_message(self.log_filepath, f"Student Architecture: {model.architecture}")
        log_message(self.log_filepath, f"Feature Dim: {model.feature_dim}")
        log_message(self.log_filepath, f"Projection Dim: {model.projector.last_layer.out_features}")
        log_message(self.log_filepath, f"Base learning rate: {base_lr}")
        log_message(self.log_filepath, f"EMA alpha: {self.ema_alpha}")
        log_message(self.log_filepath, f"Log file: {log_filepath}")

    @torch.no_grad()
    def _update_teacher(self, momentum):
        for param_s, param_t in zip(self.student_model.backbone.parameters(), self.teacher_model.backbone.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)
        for param_s, param_t in zip(self.student_model.projector.parameters(), self.teacher_model.projector.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=False)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def _dino_loss(self, student_output, teacher_output):
        s_global1, s_global2 = student_output[0], student_output[1]
        t_global1, t_global2 = teacher_output[0], teacher_output[1]
        t_softmax1 = F.softmax((t_global1 - self.center) / self.teacher_temp, dim=-1).detach()
        t_softmax2 = F.softmax((t_global2 - self.center) / self.teacher_temp, dim=-1).detach()
        total_loss = 0
        n_loss_terms = 0
        for i, s_out in enumerate(student_output):
            s_log_softmax = F.log_softmax(s_out / self.student_temp, dim=-1)
            if i == 0:
                loss = - (t_softmax2 * s_log_softmax).sum(dim=-1).mean()
            elif i == 1:
                loss = - (t_softmax1 * s_log_softmax).sum(dim=-1).mean()
            else:
                loss1 = - (t_softmax1 * s_log_softmax).sum(dim=-1).mean()
                loss2 = - (t_softmax2 * s_log_softmax).sum(dim=-1).mean()
                loss = (loss1 + loss2) / 2
            total_loss += loss
            n_loss_terms += 1
        if n_loss_terms > 0:
            return total_loss / n_loss_terms
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train(self, train_loader, epochs):
        self.student_model.train()
        self.teacher_model.train()
        loss_history = []
        best_loss = float('inf')
        no_improve = 0
        log_message(self.log_filepath, f"\nStarting DINO training for {epochs} epochs")
        warmup_epochs = 10
        lr = self.base_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
        for epoch in range(epochs):
            epoch_loss = 0
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr * lr_scale
            elif epoch == warmup_epochs:
                 for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            momentum = 1 - (1 - self.ema_alpha) * (math.cos(math.pi * epoch / epochs) + 1) / 2
            log_message(self.log_filepath, f"\nEpoch {epoch+1}/{epochs} - LR: {self.optimizer.param_groups[0]['lr']:.6f} - Teacher EMA: {momentum:.4f}")
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (views, _) in enumerate(batch_iterator):
                views = [v.to(self.device) for v in views]
                global_views = views[:2]
                with torch.no_grad():
                    t_global1 = self.teacher_model(global_views[0], pretrain_mode=True)
                    t_global2 = self.teacher_model(global_views[1], pretrain_mode=True)
                    teacher_output_globals = [t_global1, t_global2]
                    self._update_center(torch.cat(teacher_output_globals, dim=0))
                student_output = [self.student_model(v, pretrain_mode=True) for v in views]
                loss = self._dino_loss(student_output, teacher_output_globals)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_teacher(momentum)
                epoch_loss += loss.item()
                batch_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
            if epoch >= warmup_epochs:
                scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_epoch_loss)
            if (best_loss - avg_epoch_loss) > self.early_stopping_delta:
                best_loss = avg_epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            log_message(self.log_filepath,
                      f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f} | "
                      f"Best Loss: {best_loss:.4f} | Patience: {no_improve}/{self.early_stopping_patience}")
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath, f"Early stopping triggered at epoch {epoch+1}")
                break
        log_message(self.log_filepath, "\nDINO training completed")
        return loss_history

    def finetune(self, train_loader, val_loader, num_classes, epochs):
        best_val_loss = float('inf')
        no_improve = 0
        best_model_state = None
        log_message(self.log_filepath, "\nStarting fine-tuning phase")
        for param in self.student_model.backbone.parameters():
            param.requires_grad = False
        for param in self.student_model.projector.parameters():
            param.requires_grad = False
        self.student_model.add_classification_head(num_classes)
        self.student_model.classifier = self.student_model.classifier.to(self.device)
        optimizer = torch.optim.AdamW(self.student_model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        for epoch in range(epochs):
            self.student_model.train()
            total_train_loss = 0
            train_iterator = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{epochs} Train", leave=False)
            for batch_idx, (x, y) in enumerate(train_iterator):
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.student_model(x, pretrain_mode=False)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
            avg_train_loss = total_train_loss / len(train_loader)
            self.student_model.eval()
            total_val_loss = 0
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                val_iterator = tqdm(val_loader, desc=f"Finetune Epoch {epoch+1}/{epochs} Val", leave=False)
                for x_val, y_val in val_iterator:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    logits_val = self.student_model(x_val, pretrain_mode=False)
                    loss_val = criterion(logits_val, y_val)
                    total_val_loss += loss_val.item()
                    preds_val = torch.argmax(logits_val, dim=1).cpu().numpy()
                    y_true_val.extend(y_val.cpu().numpy())
                    y_pred_val.extend(preds_val)
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
            scheduler.step(avg_val_loss)
            log_message(self.log_filepath,
                       f"Finetune Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            if (best_val_loss - avg_val_loss) > self.early_stopping_delta:
                best_val_loss = avg_val_loss
                no_improve = 0
                best_model_state = copy.deepcopy(self.student_model.state_dict())
                log_message(self.log_filepath, f"  New best validation loss: {best_val_loss:.4f}. Saving model state.")
            else:
                no_improve += 1
            log_message(self.log_filepath,
                       f"  Best Val Loss: {best_val_loss:.4f} | Patience: {no_improve}/{self.early_stopping_patience}")
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath, f"Early stopping triggered at epoch {epoch+1}")
                break
        log_message(self.log_filepath, "\nFine-tuning completed")
        if best_model_state:
            self.student_model.load_state_dict(best_model_state)
            log_message(self.log_filepath, "Loaded best model state from validation.")
        else:
             log_message(self.log_filepath, "Warning: No best model state saved. Using last state.")
        return avg_val_loss

    def evaluate(self, test_loader, num_classes, class_names=None):
        if self.student_model.classifier is None:
            log_message(self.log_filepath, "Error: Classifier not found. Run fine-tuning first.")
            return None, {"error": "Classifier not initialized"}
        self.student_model.eval()
        y_true, y_pred = [], []
        log_message(self.log_filepath, "\nStarting evaluation on test set")
        with torch.no_grad():
            test_iterator = tqdm(test_loader, desc="Evaluating", leave=False)
            for x, y in test_iterator:
                x = x.to(self.device)
                logits = self.student_model(x, pretrain_mode=False)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_pred.extend(preds)
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        if num_classes == 2:
            f1_positive = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
            metrics['f1_positive'] = f1_positive
            log_message(self.log_filepath, f"Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, F1 Positive: {f1_positive:.4f}")
        else:
            log_message(self.log_filepath, f"Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
            if class_names and len(class_names) == num_classes:
                 f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                 for i, name in enumerate(class_names):
                     metrics[f'f1_{name}'] = f1_per_class[i]
                     log_message(self.log_filepath, f"  F1 {name}: {f1_per_class[i]:.4f}")
        return self.student_model.state_dict(), metrics


