import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os
import timm

EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 10)) # Aumentei um pouco a paciência
EARLY_STOPPING_DELTA = float(os.getenv("EARLY_STOPPING_DELTA", 0.001))

def log_message(log_filepath, message):
    """Função para registrar mensagens em um arquivo de log e no console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_filepath, "a") as log_file:
        log_file.write(full_message + "\n")

class MultiCropTransform:
    """Cria múltiplas views (crops) de uma imagem, conforme a metodologia DINO."""
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
    """Cabeça de projeção DINO."""
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # Usando a arquitetura exata do paper, com bottleneck
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False # Congela a norma

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2) # Normalização L2
        x = self.last_layer(x)
        return x

class DINO(nn.Module):
    """Modelo DINO completo (backbone + projetor)."""
    def __init__(self, architecture='resnet18', out_dim=65536):
        super().__init__()
        self.architecture = architecture
        try:
            if architecture in models.__dict__:
                self.backbone = models.__dict__[architecture](weights='DEFAULT')
                self.in_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                self.backbone = timm.create_model(architecture, pretrained=True, num_classes=0, global_pool='avg')
                self.in_dim = self.backbone.num_features
        except Exception as e:
            raise ValueError(f"Erro ao carregar arquitetura '{architecture}': {e}")

        
        self.projector = DINOHead(self.in_dim, out_dim=out_dim)
        self.classifier = None # Será adicionado no fine-tuning

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        
        # Se for para fine-tuning e o classificador existir, use as features do backbone
        if self.classifier is not None and not return_features:
            return self.classifier(features)
        
        # Caso contrário (treino SSL ou se explicitamente pedido), retorne a projeção
        projected = self.projector(features)
        if return_features:
            return features, projected
        return projected

class DINOLoss(nn.Module):
    """Função de perda do DINO."""
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_sm = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_output]
        
        # Aplica sharpening e centering na saída do teacher
        teacher_sm_centered = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_output]
        
        total_loss = 0
        n_loss_terms = 0
        for i, t_sm in enumerate(teacher_sm_centered):
            for j, s_lsm in enumerate(student_sm):
                # Ignora a perda da mesma view (ex: student global1 vs teacher global1)
                if i == j:
                    continue
                loss = -torch.sum(t_sm * s_lsm, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Atualiza o centro de forma exponencial (EMA)."""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOTrainer:
    def __init__(self, model, device='cuda', base_lr=0.0005, 
                 log_filepath='dino_training.log',
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 early_stopping_delta=EARLY_STOPPING_DELTA, out_dim=65536):
        self.model = model.to(device)
        self.device = device
        self.log_filepath = log_filepath
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        
        # Inicializa o teacher (não será treinado com backpropagation)
        self.teacher = DINO(model.architecture, out_dim=out_dim).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.teacher.requires_grad_(False)
        
        self.dino_loss = DINOLoss(out_dim=out_dim).to(device)
        
        # Otimizador para o student
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr)
        
        log_message(self.log_filepath, "DINO Trainer initialized")
        log_message(self.log_filepath, f"Using device: {device}")

    @torch.no_grad()
    def _update_teacher(self, momentum):
        """Atualiza os pesos do teacher via EMA (Exponential Moving Average)."""
        for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def train(self, train_loader, num_epochs, warmup_epochs=10, final_ema=1.0, initial_ema=0.996):
        self.model.train()
        self.teacher.train()
        
        # Scheduler de momento (EMA) e learning rate
        momentum_schedule = np.concatenate((
            np.linspace(initial_ema, final_ema, len(train_loader) * num_epochs),
        ))
        
        best_loss = float('inf')
        no_improve = 0
        
        log_message(self.log_filepath, f"\nStarting DINO pre-training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for i, (views, _) in enumerate(train_loader):
                # Atualiza o momento do teacher
                it = len(train_loader) * epoch + i
                momentum = momentum_schedule[it]
                self._update_teacher(momentum)

                # Move as views para o device
                views = [v.to(self.device) for v in views]
                global_views = views[:2]
                all_views = views

                # Forward pass do teacher (apenas nas views globais e sem gradientes)
                with torch.no_grad():
                    teacher_output = [self.teacher(v) for v in global_views]
                
                # Forward pass do student (em todas as views)
                student_output = [self.model(v) for v in all_views]
                
                # Calcula a perda
                loss = self.dino_loss(student_output, teacher_output)
                
                # Otimização
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            
            log_message(self.log_filepath, 
                        f"Pre-training Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} | "
                        f"Teacher EMA: {momentum:.4f}")

            # Early stopping check
            if (best_loss - avg_epoch_loss) > self.early_stopping_delta:
                best_loss = avg_epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath, f"Early stopping pre-training at epoch {epoch+1}")
                break
        
        log_message(self.log_filepath, "\nDINO pre-training completed")

    def fine_tune(self, train_loader, num_classes, epochs):
        log_message(self.log_filepath, "\nStarting fine-tuning phase (linear probing)")
        
        # Congela o backbone e o projetor
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Adiciona um novo classificador linear e o torna treinável
        self.model.classifier = nn.Linear(self.model.in_dim, num_classes).to(self.device)
        
        # Otimizador treinará apenas o novo classificador
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train() # Ativa o modo de treino para o classificador (dropout, etc.)
        
        best_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0
            
            for x, y in train_loader:
                if x.dim() == 5:
                    input_image = x[:, 0].to(self.device)
                else: # Caso o dataset já retorne 4D, o código continua funcionando
                    input_image = x.to(self.device)
                # CORREÇÃO: Trata a entrada como um tensor único
                y = y.to(self.device)
                
                logits = self.model(input_image)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            log_message(self.log_filepath,
                        f"Fine-tuning Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Early stopping check
            if (best_loss - avg_loss) > self.early_stopping_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.early_stopping_patience:
                log_message(self.log_filepath, f"Early stopping fine-tuning at epoch {epoch+1}")
                break
        
        log_message(self.log_filepath, "\nFine-tuning completed")

    def evaluate(self, test_loader, num_classes):
        self.model.eval()
        y_true, y_pred = [], []
        
        log_message(self.log_filepath, "\nStarting evaluation on test set")
        
        with torch.no_grad():
            for x, y in test_loader:
                if x.dim() == 5:
                    input_image = x[:, 0].to(self.device)
                else:
                    input_image = x.to(self.device)
                
                logits = self.model(input_image)
                preds = torch.argmax(logits, dim=1).cpu()
                
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())
                
        # --- MÉTRICAS ATUALIZADAS ---
        accuracy = accuracy_score(y_true, y_pred)
        # F1-Micro (global, equivalente à acurácia)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        # F1-Score por classe
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        log_message(self.log_filepath, f"Accuracy (F1-Micro): {f1_micro:.4f}")
        # Exibe o F1 de cada classe
        for i, f1 in enumerate(f1_per_class):
            log_message(self.log_filepath, f"  - F1-Score Classe {i}: {f1:.4f}")
        log_message(self.log_filepath, f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }

# Função de salvamento de métricas (ajustada para garantir que o diretório exista)
def save_metrics_to_txt(metrics_dict, filename, log_filepath):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for arch, arch_results in metrics_dict.items():
            f.write(f"{arch}:\n")
            for class_name, metrics in arch_results.items():
                f.write(f"  {class_name}:\n")
                f.write(f"    Accuracy: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}\n")
                f.write(f"    F1 Macro: {metrics['avg_f1_macro']:.4f} ± {metrics['std_f1_macro']:.4f}\n")
                f.write(f"    F1 Positive: {metrics['avg_f1_positive']:.4f} ± {metrics['std_f1_positive']:.4f}\n")
    log_message(log_filepath, f"Metrics saved to {filename}")