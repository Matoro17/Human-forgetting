import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import math
import os

from utils.logging_utils import log_message
from utils.metrics_utils import calculate_metrics
from utils.transforms import get_eval_transform # For fine-tuning dataset
from torch.utils.data import DataLoader, Subset

# Default hyperparameters (can be overridden)
DEFAULT_PRETRAIN_EPOCHS = int(os.getenv("SIMCLR_PRETRAIN_EPOCHS", 50))
DEFAULT_FINETUNE_EPOCHS = int(os.getenv("SIMCLR_FINETUNE_EPOCHS", 50))
DEFAULT_PRETRAIN_LR = float(os.getenv("SIMCLR_PRETRAIN_LR", 1e-3))
DEFAULT_FINETUNE_LR = float(os.getenv("SIMCLR_FINETUNE_LR", 1e-4))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEFAULT_TEMPERATURE = float(os.getenv("SIMCLR_TEMPERATURE", 0.5))
DEFAULT_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))
DEFAULT_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

class SimCLRTrainer:
    """Trainer for the SimCLR model.

    Handles both the pre-training phase using NT-Xent loss and
    the fine-tuning phase using cross-entropy loss.
    Includes early stopping.
    """
    def __init__(self, model, device="cuda", log_filepath="simclr_training.log",
                 pretrain_lr=DEFAULT_PRETRAIN_LR, finetune_lr=DEFAULT_FINETUNE_LR,
                 temperature=DEFAULT_TEMPERATURE,
                 early_stopping_patience=DEFAULT_PATIENCE,
                 early_stopping_delta=DEFAULT_DELTA):

        self.model = model.to(device)
        self.device = device
        self.log_filepath = log_filepath
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.temperature = temperature
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.pretrain_time = 0.0
        self.finetune_time = 0.0
        self.pretrain_loss_history = []
        self.finetune_loss_history = []

        log_message(self.log_filepath, f"SimCLR Trainer initialized on device: {self.device}")
        log_message(self.log_filepath, f"Pretrain LR: {self.pretrain_lr}, Finetune LR: {self.finetune_lr}, Temp: {self.temperature}")
        log_message(self.log_filepath, f"Early Stopping Patience: {self.early_stopping_patience}, Delta: {self.early_stopping_delta}")

    def pretrain(self, train_loader, epochs=DEFAULT_PRETRAIN_EPOCHS):
        """Performs SimCLR pre-training."""
        log_message(self.log_filepath, f"\n--- Starting SimCLR Pre-training for {epochs} epochs ---")
        self.model.train()
        # Ensure classification head is removed/ignored during pre-training
        self.model.classification_head = None

        optimizer = optim.Adam(self.model.parameters(), lr=self.pretrain_lr)
        # Simple scheduler: decay LR by factor of 10 halfway through
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 2, gamma=0.1)

        best_loss = float("inf")
        no_improve_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}", leave=False)
            for views, _ in pbar:
                # views should be a tuple/list of two tensors from SimCLRTransform
                if isinstance(views, list) or isinstance(views, tuple):
                    view1, view2 = views[0].to(self.device), views[1].to(self.device)
                else: # Handle case where transform might return stacked tensor
                     if views.ndim == 5 and views.shape[1] == 2: # Shape [B, 2, C, H, W]
                         view1, view2 = views[:, 0].to(self.device), views[:, 1].to(self.device)
                     else:
                         log_message(self.log_filepath, f"Unexpected input shape in pretrain: {views.shape}. Skipping batch.")
                         continue

                optimizer.zero_grad()

                # Forward pass - pretrain=True returns features and projections
                _, z1 = self.model(view1, pretrain=True)
                _, z2 = self.model(view2, pretrain=True)

                loss = self.model.nt_xent_loss(z1, z2, temperature=self.temperature)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.pretrain_loss_history.append(avg_loss)
            epoch_time = time.time() - epoch_start_time
            log_message(self.log_filepath, f"Pretrain Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

            scheduler.step()

            # Early stopping check
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
        log_message(self.log_filepath, f"--- SimCLR Pre-training finished. Total time: {self.pretrain_time:.2f}s ---")
        return self.pretrain_loss_history

    def finetune(self, train_loader, val_loader, num_classes, epochs=DEFAULT_FINETUNE_EPOCHS):
        """Performs fine-tuning on a classification task."""
        log_message(self.log_filepath, f"\n--- Starting SimCLR Fine-tuning for {epochs} epochs ---")

        self.model.add_classification_head(num_classes)
        self.model.to(self.device)

        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for param in self.model.projection_head.parameters():
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
        for name, param in self.model.backbone.named_parameters():
            if "layer4" in name or "fc" in name:
                 param.requires_grad = True
            else:
                 param.requires_grad = False

        optimizer = optim.Adam([
            {"params": filter(lambda p: p.requires_grad, self.model.backbone.parameters()), "lr": self.finetune_lr / 10},
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
        log_message(self.log_filepath, f"--- SimCLR Fine-tuning finished. Total time: {self.finetune_time:.2f}s ---")
        return self.finetune_loss_history

    def evaluate(self, test_loader, num_classes, class_names=None, validation_mode=False):
        """Evaluates the fine-tuned model on the test set."""
        if not validation_mode:
             log_message(self.log_filepath, f"\n--- Evaluating SimCLR on Test Set ---")
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

