import torch
import torch.nn as nn
import copy

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
