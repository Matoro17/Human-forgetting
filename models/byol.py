import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import os

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class BYOL(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(BYOL, self).__init__()
        # Online network
        self.online_encoder = encoder
        self.online_projection = MLPHead(512, projection_dim)
        self.online_predictor = MLPHead(projection_dim, projection_dim)

        # Target network (initialized as a copy of the online network)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projection = copy.deepcopy(self.online_projection)

        # Freeze the target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projection.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        # Online network forward pass
        online_proj1 = self.online_projection(self.online_encoder(x1))
        online_proj2 = self.online_projection(self.online_encoder(x2))
        pred1 = self.online_predictor(online_proj1)
        pred2 = self.online_predictor(online_proj2)

        # Target network forward pass (no gradients)
        with torch.no_grad():
            target_proj1 = self.target_projection(self.target_encoder(x1))
            target_proj2 = self.target_projection(self.target_encoder(x2))

        return pred1, pred2, target_proj1.detach(), target_proj2.detach()

class BYOLTrainer:
    def __init__(self, model, device='cuda', lr=1e-3, ema_decay=0.996, log_filepath='byol_training.log'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.log_filepath = log_filepath
        self.early_stopping_patience = 10
        self.early_stopping_delta = 0.001

    def _update_target_network(self):
        for online_params, target_params in zip(
            list(self.model.online_encoder.parameters()) + list(self.model.online_projection.parameters()),
            list(self.model.target_encoder.parameters()) + list(self.model.target_projection.parameters())
        ):
            target_params.data = self.ema_decay * target_params.data + (1. - self.ema_decay) * online_params.data

    def train(self, train_loader, epochs):
        self.model.train()
        best_loss = float('inf')
        no_improve = 0
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for (x, _), in train_loader:
                x1 = x[:, 0].to(self.device)
                x2 = x[:, 1].to(self.device)

                pred1, pred2, target1, target2 = self.model(x1, x2)

                loss = loss_fn(pred1, target2) + loss_fn(pred2, target1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[BYOL] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if avg_loss < best_loss - self.early_stopping_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    print("Early stopping BYOL pretraining.")
                    break

    def fine_tune(self, train_loader, num_classes, epochs):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Linear(512, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y in train_loader:
                x = x[:, 0].to(self.device)
                y = y.to(self.device)
                features = self.model.online_encoder(x)
                logits = self.model.classifier(features)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[BYOL] Fine-tune Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if avg_loss < best_loss - self.early_stopping_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    print("Early stopping fine-tuning.")
                    break

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x = x[:, 0].to(self.device)
                features = self.model.online_encoder(x)
                logits = self.model.classifier(features)
                preds = torch.argmax(logits, dim=1).cpu()
                y_true.extend(y.numpy())
                y_pred.extend(preds.numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_positive = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)

        print(f"[BYOL] Evaluation - Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, F1 Positive: {f1_positive:.4f}")
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_positive': f1_positive,
            'confusion_matrix': cm
        }
