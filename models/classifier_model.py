import torch.nn as nn
from models.base_model import load_base_model

class CIFAR10Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Classifier, self).__init__()
        base_model = load_base_model()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)
        self.base_model.fc = self.classifier

    def forward(self, x):
        return self.base_model(x)