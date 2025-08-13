# dino.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os
import timm
from tqdm import tqdm

# --- CONFIGURATIONS ---
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 15))
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

# --- HELPER FUNCTIONS ---

def log_message(log_filepath, message):
    """Logs a message to a file and prints it to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    if log_filepath:
        with open(log_filepath, "a") as log_file:
            log_file.write(full_message + "\n")

# --- CORE DINO COMPONENTS ---

class MultiCropTransform:
    """
    Creates multiple views (crops) of an image, as used in DINO.
    - 2 global crops at a higher resolution.
    - Multiple local crops at a lower resolution.
    """
    def __init__(self, global_size=224, local_size=96, num_local=4):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
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
    """
    DINO projection head. Takes features from the backbone and projects
    them into a high-dimensional space.
    """
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
        x = nn.functional.normalize(x, dim=-1, p=2) # L2 normalization
        x = self.last_layer(x)
        return x

class DINO(nn.Module):
    """
    Main DINO model, combining a backbone with the projection head.
    The backbone can be any ViT or Swin model from the 'timm' library.
    """
    def __init__(self, architecture, out_dim=65536, use_pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            architecture,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.in_dim = self.backbone.num_features

        self.projector = DINOHead(self.in_dim, out_dim=out_dim)
        self.classifier = None # Linear classifier for fine-tuning

    def forward(self, x):
        features = self.backbone(x)
        if self.classifier is not None:
            return self.classifier(features)
        
        return self.projector(features)

class DINOLoss(nn.Module):
    """
    The core loss function for DINO. It computes the cross-entropy
    between the student's predictions and the centered, sharpened 
    outputs of the teacher.
    """
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_sm = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_output]
        teacher_sm_centered = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_output]
        
        total_loss = 0
        n_loss_terms = 0
        for i, t_sm in enumerate(teacher_sm_centered):
            for j, s_lsm in enumerate(student_sm):
                if i == j: # Skip comparing a view with itself
                    continue
                loss = -torch.sum(t_sm * s_lsm, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update the center with exponential moving average."""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOTrainer:
    """
    Orchestrates the DINO training process, including pre-training and fine-tuning.
    """
    def __init__(self, model, device='cuda', base_lr=0.0005, log_filepath='dino_training.log', out_dim=65536):
        self.device = device
        self.log_filepath = log_filepath

        self.model = model.to(device)
        self.teacher = DINO(model.backbone.default_cfg['architecture'], out_dim=out_dim).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.teacher.requires_grad_(False)
        
        self.dino_loss = DINOLoss(out_dim=out_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr)
        
        log_message(self.log_filepath, "DINO Trainer initialized")
        log_message(self.log_filepath, f"Using device: {device}")

    @torch.no_grad()
    def _update_teacher(self, momentum):
        """Update teacher weights with an EMA of student weights."""
        for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def train(self, train_loader, num_epochs, warmup_epochs=10, final_ema=1.0, initial_ema=0.996):
        self.model.train()
        self.teacher.train()
        
        total_steps = len(train_loader) * num_epochs
        momentum_schedule = np.array([final_ema - (final_ema - initial_ema) * (math.cos(math.pi * i / total_steps) + 1) / 2 for i in range(total_steps)])
        
        best_loss = float('inf')
        no_improve = 0
        
        log_message(self.log_filepath, f"\nStarting DINO pre-training for {num_epochs} epochs")
        
        model_input_size = self.model.backbone.patch_embed.img_size[0]

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            
            for i, (views, _) in enumerate(train_loader):
                it = len(train_loader) * epoch + i
                momentum = momentum_schedule[it]
                self._update_teacher(momentum)

                views = [v.to(self.device, non_blocking=True) for v in views]
                
                global_views = views[:2] 
                local_views = views[2:]

                with torch.no_grad():
                    teacher_output = [self.teacher(v) for v in global_views]
                
                # **FIXED**: The `lv` tensor is already a batch (N,C,H,W), so no unsqueeze is needed.
                resized_local_views = [F.interpolate(lv, size=(model_input_size, model_input_size), mode='bicubic', align_corners=False) for lv in local_views]
                all_views_for_student = global_views + resized_local_views

                student_output = [self.model(v) for v in all_views_for_student]
                
                loss = self.dino_loss(student_output, teacher_output)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())


            avg_epoch_loss = epoch_loss / len(train_loader)
            log_message(self.log_filepath, f"Pre-training Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} | Teacher EMA: {momentum:.4f}")

            if (best_loss - avg_epoch_loss) > EARLY_STOPPING_DELTA:
                best_loss = avg_epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= EARLY_STOPPING_PATIENCE:
                log_message(self.log_filepath, f"Early stopping pre-training at epoch {epoch+1}")
                break
        
        log_message(self.log_filepath, "\nDINO pre-training completed")

    def fine_tune(self, train_loader, num_classes, epochs):
        log_message(self.log_filepath, "\nStarting fine-tuning phase (linear probing)")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = nn.Linear(self.model.in_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            for x, y in train_loader:
                input_image = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(input_image)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            log_message(self.log_filepath, f"Fine-tuning Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        log_message(self.log_filepath, "\nFine-tuning completed")

    def evaluate(self, test_loader, num_classes):
        self.model.eval()
        y_true, y_pred = [], []
        
        log_message(self.log_filepath, "\nStarting evaluation on test set")
        
        with torch.no_grad():
            for x, y in test_loader:
                input_image = x.to(self.device)
                logits = self.model(input_image)
                preds = torch.argmax(logits, dim=1).cpu()
                
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        log_message(self.log_filepath, f"Accuracy (F1-Micro): {f1_micro:.4f}")
        for i, f1 in enumerate(f1_per_class):
            log_message(self.log_filepath, f"  - F1-Score Class {i}: {f1:.4f}")
        log_message(self.log_filepath, f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }