import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

# 1. SimCLR Model Architecture
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

class ProjectionHead(nn.Module):
    def __init__(self, encoder):
        super(ProjectionHead, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Fine-Tune Model Definition
class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
