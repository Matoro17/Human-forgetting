import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import math
import os
import numpy as np

from utils.logging_utils import log_message
from utils.metrics_utils import calculate_metrics
from utils.transforms import get_eval_transform # For fine-tuning dataset
from torch.utils.data import DataLoader, Subset

# Default hyperparameters (can be overridden)
DEFAULT_PRETRAIN_EPOCHS = int(os.getenv("DINO_PRETRAIN_EPOCHS", 50))
DEFAULT_FINETUNE_EPOCHS = int(os.getenv("DINO_FINETUNE_EPOCHS", 50))
DEFAULT_PRETRAIN_LR = float(os.getenv("DINO_PRETRAIN_LR", 5e-4)) # DINO often uses AdamW
DEFAULT_FINETUNE_LR = float(os.getenv("DINO_FINETUNE_LR", 1e-4))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEFAULT_WEIGHT_DECAY = float(os.getenv("DINO_WEIGHT_DECAY", 0.04))
DEFAULT_WEIGHT_DECAY_END = float(os.getenv("DINO_WEIGHT_DECAY_END", 0.4))
DEFAULT_STUDENT_TEMP = float(os.getenv("DINO_STUDENT_TEMP", 0.1))
DEFAULT_TEACHER_TEMP_WARMUP = float(os.getenv("DINO_TEACHER_TEMP_WARMUP", 0.04))
DEFAULT_TEACHER_TEMP = float(os.getenv("DINO_TEACHER_TEMP", 0.07))
DEFAULT_TEACHER_TEMP_WARMUP_EPOCHS = int(os.getenv("DINO_TEACHER_TEMP_WARMUP_EPOCHS", 30))
DEFAULT_BASE_MOMENTUM = float(os.getenv("DINO_BASE_MOMENTUM", 0.996))
DEFAULT_END_MOMENTUM = float(os.getenv("DINO_END_MOMENTUM", 1.0))
DEFAULT_CENTER_MOMENTUM = float(os.getenv("DINO_CENTER_MOMENTUM", 0.9))
DEFAULT_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
DEFAULT_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

class DINOTrainer:
    """Trainer for the DINO model.

    Handles the pre-training phase using DINO loss (self-distillation w/ no labels)
    and the fine-tuning phase using cross-entropy loss.
    Includes EMA updates for the teacher network, center update, and early stopping.
    """
    def __init__(self, model, device="cuda", log_filepath="dino_training.log",
                 pretrain_lr=DEFAULT_PRETRAIN_LR, finetune_lr=DEFAULT_FINETUNE_LR,
                 weight_decay=DEFAULT_WEIGHT_DECAY, weight_decay_end=DEFAULT_WEIGHT_DECAY_END,
                 student_temp=DEFAULT_STUDENT_TEMP, teacher_temp_warmup=DEFAULT_TEACHER_TEMP_WARMUP,
                 teacher_temp=DEFAULT_TEACHER_TEMP, teacher_temp_warmup_epochs=DEFAULT_TEACHER_TEMP_WARMUP_EPOCHS,
                 base_momentum=DEFAULT_BASE_MOMENTUM, end_momentum=DEFAULT_END_MOMENTUM,
                 center_momentum=DEFAULT_CENTER_MOMENTUM,
                 early_stopping_patience=DEFAULT_PATIENCE,
                 early_stopping_delta=DEFAULT_DELTA):

        self.model = model.to(device)
        self.device = device
        self.log_filepath = log_filepath
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end
        self.student_temp = student_temp
        self.teacher_temp_warmup = teacher_temp_warmup
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.base_momentum = base_momentum
        self.end_momentum = end_momentum
        self.center_momentum = center_momentum
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.register_buffer("center", torch.zeros(1, model.out_dim, device=device), persistent=False)

        self.pretrain_time = 0.0
        self.finetune_time = 0.0
        self.pretrain_loss_history = []
        self.finetune_loss_history = []

        log_message(self.log_filepath, f"DINO Trainer initialized on device: {self.device}")
        log_message(self.log_filepath, f"Pretrain LR: {self.pretrain_lr}, Finetune LR: {self.finetune_lr}")
        log_message(self.log_filepath, f"WD Schedule: {self.weight_decay} -> {self.weight_decay_end}")
        log_message(self.log_filepath, f"Momentum Schedule: {self.base_momentum} -> {self.end_momentum}")
        log_message(self.log_filepath, f"Temperatures: Student={self.student_temp}, Teacher={self.teacher_temp} (Warmup={self.teacher_temp_warmup} for {self.teacher_temp_warmup_epochs} epochs)")
        log_message(self.log_filepath, f"Center Momentum: {self.center_momentum}")
        log_message(self.log_filepath, f"Early Stopping Patience: {self.early_stopping_patience}, Delta: {self.early_stopping_delta}")

    def register_buffer(self, name, tensor, persistent=True):
        setattr(self, name, tensor)

    def _calculate_momentum(self, current_epoch, total_epochs):
        return self.end_momentum - (self.end_momentum - self.base_momentum) * (math.cos(math.pi * current_epoch / total_epochs) + 1) / 2

    def _calculate_teacher_temp(self, current_epoch):
        if current_epoch < self.teacher_temp_warmup_epochs:
            return self.teacher_temp_warmup + (self.teacher_temp - self.teacher_temp_warmup) * current_epoch / self.teacher_temp_warmup_epochs
        else:
            return self.teacher_temp

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def pretrain(self, train_loader, epochs=DEFAULT_PRETRAIN_EPOCHS):
        log_message(self.log_filepath, f"\n--- Starting DINO Pre-training for {epochs} epochs ---")
        self.model.train()
        self.model.classification_head = None

        optimizer = optim.AdamW(self.model.parameters(), lr=self.pretrain_lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        wd_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.weight_decay + (self.weight_decay_end - self.weight_decay) * (1 + math.cos(math.pi * epoch / epochs)) / 2)

        best_loss = float("inf")
        no_improve_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}", leave=False)
            for views, _ in pbar:
                if not isinstance(views, list):
                    log_message(self.log_filepath, f"Input must be a list of views for DINO. Skipping batch.")
                    continue
                views = [v.to(self.device) for v in views]

                optimizer.zero_grad()
                student_output, teacher_output = self.model(views, pretrain=True)
                current_teacher_temp = self._calculate_teacher_temp(epoch)
                loss = self.model.dino_loss(student_output, teacher_output, self.student_temp, current_teacher_temp, self.center)
                loss.backward()
                optimizer.step()

                current_momentum = self._calculate_momentum(epoch, epochs)
                self.model.update_teacher_network(current_momentum)
                self._update_center(teacher_output)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "t_temp": f"{current_teacher_temp:.4f}", "mom": f"{current_momentum:.4f}"}) 

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.pretrain_loss_history.append(avg_loss)
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            log_message(self.log_filepath, f"Pretrain Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, WD: {current_wd:.4f}, T_Temp: {current_teacher_temp:.4f}, Momentum: {current_momentum:.4f}, Time: {epoch_time:.2f}s")

            lr_scheduler.step()
            wd_scheduler.step()

            if avg_loss < best_loss - self.early_stopping_delta:
                best_loss = avg_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                log_message(self.log_filepath, f"Early stopping counter: {no_improve_counter}/{self.early_stopping_patience}")
                if no_improve_counter >= self.early_stopping_patience:
                    log_message(self.log_filepath, f"Early stopping triggered at epoch {epoch+1}")
                    break

        self.pretrain_time = time.time() - start_time
        log_message(self.log_filepath, f"--- DINO Pre-training finished. Total time: {self.pretrain_time:.2f}s ---")
        return self.pretrain_loss_history

    def finetune(self, train_loader, val_loader, num_classes, epochs=DEFAULT_FINETUNE_EPOCHS):
        log_message(self.log_filepath, f"\n--- Starting DINO Fine-tuning for {epochs} epochs ---")

        self.model.add_classification_head(num_classes)
        self.model.to(self.device)

        for param in self.model.student_backbone.parameters():
            param.requires_grad = False
        for param in self.model.student_head.parameters():
             param.requires_grad = False

        optimizer = optim.Adam(self.model.classification_head.parameters(), lr=self.finetune_lr * 10)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)

        best_val_loss = float("inf")
        no_improve_counter = 0
        start_time = time.time()

        log_message(self.log_filepath, "Fine-tuning Phase 1: Training Classifier Head")
        num_classifier_epochs = max(5, epochs // 5)
        for epoch in range(num_classifier_epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Finetune (CLS) Epoch {epoch+1}/{num_classifier_epochs}", leave=False)
            for inputs, labels in pbar:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = inputs[0].to(self.device)
                elif inputs.ndim == 5:
                    inputs = inputs[:, 0].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs, pretrain=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time
            log_message(self.log_filepath, f"Finetune (CLS) Epoch {epoch+1}/{num_classifier_epochs} - Avg Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
            scheduler.step()

        log_message(self.log_filepath, "Fine-tuning Phase 2: Unfreezing Backbone Layers")
        for name, param in self.model.student_backbone.named_parameters():
            if "layer4" in name or "fc" in name:
                 param.requires_grad = True
            else:
                 param.requires_grad = False

        optimizer = optim.Adam([
            {"params": filter(lambda p: p.requires_grad, self.model.student_backbone.parameters()), "lr": self.finetune_lr / 10},
            {"params": self.model.classification_head.parameters(), "lr": self.finetune_lr}
        ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=(epochs - num_classifier_epochs) // 3, gamma=0.1)

        remaining_epochs = epochs - num_classifier_epochs
        for epoch in range(remaining_epochs):
            epoch_idx = epoch + num_classifier_epochs
            epoch_start_time = time.time()
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Finetune (Full) Epoch {epoch_idx+1}/{epochs}", leave=False)
            for inputs, labels in pbar:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = inputs[0].to(self.device)
                elif inputs.ndim == 5:
                    inputs = inputs[:, 0].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs, pretrain=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            self.finetune_loss_history.append(avg_train_loss)

            avg_val_loss, _ = self.evaluate(val_loader, num_classes, validation_mode=True)
            epoch_time = time.time() - epoch_start_time
            log_message(self.log_filepath, f"Finetune Epoch {epoch_idx+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

            scheduler.step()

            if avg_val_loss < best_val_loss - self.early_stopping_delta:
                best_val_loss = avg_val_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                log_message(self.log_filepath, f"Early stopping counter: {no_improve_counter}/{self.early_stopping_patience}")
                if no_improve_counter >= self.early_stopping_patience:
                    log_message(self.log_filepath, f"Early stopping triggered at epoch {epoch_idx+1}")
                    break

        self.finetune_time = time.time() - start_time
        log_message(self.log_filepath, f"--- DINO Fine-tuning finished. Total time: {self.finetune_time:.2f}s ---")
        return self.finetune_loss_history

    def evaluate(self, test_loader, num_classes, class_names=None, validation_mode=False):
        if not validation_mode:
             log_message(self.log_filepath, f"\n--- Evaluating DINO on Test Set ---")
        self.model.eval()
        y_true = []
        y_pred = []
        total_loss = 0.0
        num_batches = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating", leave=False)
            for inputs, labels in pbar:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = inputs[0].to(self.device)
                elif inputs.ndim == 5:
                    inputs = inputs[:, 0].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs, pretrain=False)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics = calculate_metrics(y_true, y_pred, num_classes, class_names)

        if not validation_mode:
            # Corrected f-string formatting below
            acc = metrics["accuracy"]
            f1_macro = metrics["f1_macro"]
            log_message(self.log_filepath, f"Evaluation Results - Loss: {avg_loss:.4f}")
            log_message(self.log_filepath, f"  Accuracy: {acc:.4f}")
            log_message(self.log_filepath, f"  F1 Macro: {f1_macro:.4f}")
            if class_names:
                for i, name in enumerate(class_names):
                    metric_key = f"f1_{name}"
                    f1_value = metrics.get(metric_key, "N/A")
                    log_message(self.log_filepath, f"  F1 {name}: {f1_value if isinstance(f1_value, str) else f'{f1_value:.4f}'}") 
            else:
                 for i in range(num_classes):
                    metric_key = f"f1_class_{i}"
                    f1_value = metrics.get(metric_key, "N/A")
                    log_message(self.log_filepath, f"  F1 Class {i}: {f1_value if isinstance(f1_value, str) else f'{f1_value:.4f}'}") 

        return avg_loss, metrics

