import torch.nn as nn

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
