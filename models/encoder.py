import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
