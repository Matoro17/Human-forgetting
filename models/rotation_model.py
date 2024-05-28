import torch.nn as nn
from models.base_model import load_base_model

class RotationPredictionModel(nn.Module):
    def __init__(self):
        super(RotationPredictionModel, self).__init__()
        base_model = load_base_model()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.fc.in_features, 4)
        self.base_model.fc = self.classifier

    def forward(self, x):
        return self.base_model(x)