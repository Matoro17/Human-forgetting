import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Adicionado import que faltava
import torch.optim as optim
from tqdm import tqdm
import time
import math
import os
import numpy as np

from utils.logging_utils import log_message # Adjusted relative import
from utils.metrics_utils import calculate_metrics # Adjusted relative import
# from utils.transforms import get_eval_transform # Not needed directly here
# from torch.utils.data import DataLoader, Subset # Not needed directly here

# Default hyperparameters (can be overridden)
DEFAULT_PRETRAIN_EPOCHS = int(os.getenv("BYOL_PRETRAIN_EPOCHS", 50))
DEFAULT_FINETUNE_EPOCHS = int(os.getenv("BYOL_FINETUNE_EPOCHS", 50))
DEFAULT_PRETRAIN_LR = float(os.getenv("BYOL_PRETRAIN_LR", 3e-3)) # BYOL often uses larger LR with LARS/AdamW
DEFAULT_FINETUNE_LR = float(os.getenv("BYOL_FINETUNE_LR", 1e-4))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEFAULT_BASE_MOMENTUM = float(os.getenv("BYOL_BASE_MOMENTUM", 0.996))
DEFAULT_END_MOMENTUM = float(os.getenv("BYOL_END_MOMENTUM", 1.0))
DEFAULT_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
DEFAULT_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

class BYOLTrainer:
    """Trainer for the BYOL model.

    Handles the pre-training phase using BYOL loss and
    the fine-tuning phase using cross-entropy loss.
    Includes EMA updates for the target network and early stopping.
    """
    def __init__(self, model, device="cuda", log_filepath="byol_training.log",
                 pretrain_lr=DEFAULT_PRETRAIN_LR, finetune_lr=DEFAULT_FINETUNE_LR,
                 base_momentum=DEFAULT_BASE_MOMENTUM, end_momentum=DEFAULT_END_MOMENTUM,
                 early_stopping_patience=DEFAULT_PATIENCE,
                 early_stopping_delta=DEFAULT_DELTA):

        self.model = model.to(device)
        self.device = device
        self.log_filepath = log_filepath
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.base_momentum = base_momentum
        self.end_momentum = end_momentum
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.pretrain_time = 0.0
        self.finetune_time = 0.0
        self.pretrain_loss_history = []
        self.finetune_loss_history = []

        log_message(self.log_filepath, f"BYOL Trainer initialized on device: {self.device}")
        log_message(self.log_filepath, f"Pretrain LR: {self.pretrain_lr}, Finetune LR: {self.finetune_lr}")
        log_message(self.log_filepath, f"Momentum Schedule: {self.base_momentum} -> {self.end_momentum}")
        log_message(self.log_filepath, f"Early Stopping Patience: {self.early_stopping_patience}, Delta: {self.early_stopping_delta}")

    def _calculate_momentum(self, current_epoch, total_epochs):
        """Calculates the EMA momentum for the current epoch based on a cosine schedule."""
        return self.end_momentum - (self.end_momentum - self.base_momentum) * (math.cos(math.pi * current_epoch / total_epochs) + 1) / 2

    def pretrain(self, train_loader, epochs=DEFAULT_PRETRAIN_EPOCHS):
        """Performs BYOL pre-training."""
        log_message(self.log_filepath, f"\n--- Starting BYOL Pre-training for {epochs} epochs ---")
        self.model.train()
        self.model.classification_head = None # Ensure no classification head during pretrain

        optimizer = optim.AdamW(self.model.parameters(), lr=self.pretrain_lr, weight_decay=1.5e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

        best_loss = float("inf")
        no_improve_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}", leave=False)
            for views, _ in pbar:
                if isinstance(views, list) or isinstance(views, tuple):
                    view1, view2 = views[0].to(self.device), views[1].to(self.device)
                else:
                     # Handle cases where DataLoader might return a stacked tensor
                     if views.ndim == 5 and views.shape[1] == 2:
                         view1, view2 = views[:, 0].to(self.device), views[:, 1].to(self.device)
                     else:
                         log_message(self.log_filepath, f"Unexpected input shape in pretrain: {views.shape}. Skipping batch.")
                         continue

                optimizer.zero_grad()
                # Pass pretrain=True to model forward
                pred1, pred2, target_proj1, target_proj2 = self.model(view1, view2, pretrain=True)
                loss = self.model.byol_loss(pred1, pred2, target_proj1, target_proj2)
                loss.backward()
                optimizer.step()

                current_momentum = self._calculate_momentum(epoch, epochs)
                self.model.update_target_network(current_momentum)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "mom": f"{current_momentum:.4f}"}) 

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.pretrain_loss_history.append(avg_loss)
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            log_message(self.log_filepath, f"Pretrain Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Momentum: {current_momentum:.4f}, Time: {epoch_time:.2f}s")

            scheduler.step()

            # Simple early stopping based on pretrain loss (optional)
            if avg_loss < best_loss - self.early_stopping_delta:
                best_loss = avg_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                log_message(self.log_filepath, f"Pretrain early stopping counter: {no_improve_counter}/{self.early_stopping_patience}")
                if no_improve_counter >= self.early_stopping_patience:
                    log_message(self.log_filepath, f"Pretrain early stopping triggered at epoch {epoch+1}")
                    break

        self.pretrain_time = time.time() - start_time
        log_message(self.log_filepath, f"--- BYOL Pre-training finished. Total time: {self.pretrain_time:.2f}s ---")
        return self.pretrain_loss_history

    def finetune(self, train_loader, val_loader, num_classes, epochs=DEFAULT_FINETUNE_EPOCHS):
        """Performs fine-tuning on a classification task."""
        log_message(self.log_filepath, f"\n--- Starting BYOL Fine-tuning for {epochs} epochs ---")

        self.model.add_classification_head(num_classes)
        self.model.to(self.device)

        # --- Phase 1: Train only the classifier head ---
        log_message(self.log_filepath, "Fine-tuning Phase 1: Training Classifier Head")
        for param in self.model.online_encoder.parameters():
            param.requires_grad = False
        for param in self.model.online_projector.parameters():
             param.requires_grad = False
        for param in self.model.online_predictor.parameters():
             param.requires_grad = False
        for param in self.model.classification_head.parameters():
            param.requires_grad = True # Ensure head is trainable

        # Use higher LR for the head initially
        optimizer_head = optim.Adam(self.model.classification_head.parameters(), lr=self.finetune_lr * 10)
        criterion = nn.CrossEntropyLoss()
        scheduler_head = optim.lr_scheduler.StepLR(optimizer_head, step_size=epochs // 3, gamma=0.1)

        num_classifier_epochs = max(5, epochs // 5) # Train head for a few epochs
        for epoch in range(num_classifier_epochs):
            epoch_start_time = time.time()
            self.model.train() # Set model to train mode (affects dropout, batchnorm)
            self.model.online_encoder.eval() # Keep backbone in eval mode if using batchnorm

            total_train_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Finetune (CLS) Epoch {epoch+1}/{num_classifier_epochs}", leave=False)
            for inputs, labels in pbar:
                # Expect eval_transform applied by DataLoader
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer_head.zero_grad()
                # Pass pretrain=False to model forward
                outputs = self.model(inputs, pretrain=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_head.step()

                total_train_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time
            log_message(self.log_filepath, f"Finetune (CLS) Epoch {epoch+1}/{num_classifier_epochs} - Avg Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
            scheduler_head.step()

        # --- Phase 2: Unfreeze backbone and train end-to-end ---
        log_message(self.log_filepath, "Fine-tuning Phase 2: Unfreezing Backbone Layers")
        for param in self.model.online_encoder.parameters():
            param.requires_grad = True # Unfreeze all backbone layers
        # Keep projector/predictor frozen if they exist
        if hasattr(self.model, 'online_projector'):
             for param in self.model.online_projector.parameters():
                 param.requires_grad = False
        if hasattr(self.model, 'online_predictor'):
             for param in self.model.online_predictor.parameters():
                 param.requires_grad = False

        # Use lower LR for backbone, keep original LR for head
        optimizer_full = optim.Adam([
            {"params": self.model.online_encoder.parameters(), "lr": self.finetune_lr / 10},
            {"params": self.model.classification_head.parameters(), "lr": self.finetune_lr}
        ])
        criterion = nn.CrossEntropyLoss()
        scheduler_full = optim.lr_scheduler.StepLR(optimizer_full, step_size=(epochs - num_classifier_epochs) // 3, gamma=0.1)

        best_val_loss = float("inf")
        no_improve_counter = 0
        start_time_full = time.time()

        remaining_epochs = epochs - num_classifier_epochs
        for epoch in range(remaining_epochs):
            epoch_idx = epoch + num_classifier_epochs
            epoch_start_time = time.time()
            self.model.train() # Set model to train mode
            total_train_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Finetune (Full) Epoch {epoch_idx+1}/{epochs}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer_full.zero_grad()
                outputs = self.model(inputs, pretrain=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_full.step()

                total_train_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
            self.finetune_loss_history.append(avg_train_loss)

            # Validation step
            avg_val_loss, _ = self.evaluate(val_loader, num_classes, validation_mode=True)
            epoch_time = time.time() - epoch_start_time
            log_message(self.log_filepath, f"Finetune Epoch {epoch_idx+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")

            scheduler_full.step()

            # Early stopping check based on validation loss
            if avg_val_loss < best_val_loss - self.early_stopping_delta:
                best_val_loss = avg_val_loss
                no_improve_counter = 0
                # torch.save(self.model.state_dict(), f"byol_finetune_best_fold_{fold_num+1}.pth") # Optional
            else:
                no_improve_counter += 1
                log_message(self.log_filepath, f"Finetune early stopping counter: {no_improve_counter}/{self.early_stopping_patience}")
                if no_improve_counter >= self.early_stopping_patience:
                    log_message(self.log_filepath, f"Finetune early stopping triggered at epoch {epoch_idx+1}")
                    break

        self.finetune_time = time.time() - start_time_full + (epoch_start_time - start_time_full) # Rough estimate
        log_message(self.log_filepath, f"--- BYOL Fine-tuning finished. Total time: {self.finetune_time:.2f}s ---")
        return self.finetune_loss_history

    def evaluate(self, test_loader, num_classes, class_names=None, validation_mode=False):
        """Evaluates the fine-tuned model on the test set or validation set."""
        if not validation_mode:
             log_message(self.log_filepath, f"\n--- Evaluating BYOL on Test Set ---")
        self.model.eval()
        y_true = []
        y_pred = []
        total_loss = 0.0
        num_batches = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            desc = "Validating" if validation_mode else "Evaluating"
            pbar = tqdm(test_loader, desc=desc, leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

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

