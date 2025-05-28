import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

class MLPHead(nn.Module):
    """MLP Head used for projection and prediction in BYOL."""
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super(MLPHead, self).__init__()
        # Original BYOL uses BN -> ReLU -> Linear
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class BYOL(nn.Module):
    """BYOL model implementation.

    Contains online and target networks (encoder, projector).
    The online network also has a predictor head.
    Target network weights are updated via EMA.
    """
    def __init__(self, backbone_name="resnet18", use_pretrained=True, projection_dim=256, projection_hidden_dim=4096):
        super(BYOL, self).__init__()
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.projection_hidden_dim = projection_hidden_dim

        # Create backbone
        self.online_encoder, self.feature_dim = self._create_backbone(backbone_name, use_pretrained)
        self.target_encoder, _ = self._create_backbone(backbone_name, use_pretrained)

        # Online network components
        self.online_projector = MLPHead(self.feature_dim, projection_hidden_dim, projection_dim)
        self.online_predictor = MLPHead(projection_dim, projection_hidden_dim, projection_dim)

        # Target network components (initialized as copy, then frozen)
        self.target_projector = MLPHead(self.feature_dim, projection_hidden_dim, projection_dim)
        self._initial_sync()
        self._freeze_target()

        # Classification head (added during fine-tuning)
        self.classification_head = None

    def _create_backbone(self, backbone_name, use_pretrained):
        """Creates the backbone network."""
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone_name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if use_pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        # Remove the original classifier
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    def _initial_sync(self):
        """Initializes target network with online network weights."""
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())

    def _freeze_target(self):
        """Freezes the target network parameters."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self, momentum):
        """Performs EMA update of the target network."""
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def forward(self, x1, x2=None, pretrain=True):
        """Forward pass.

        Args:
            x1 (torch.Tensor): First view of the image(s).
            x2 (torch.Tensor, optional): Second view of the image(s). Required if pretrain=True.
            pretrain (bool): If True, performs BYOL pre-training forward pass.
                             If False, performs classification forward pass using online encoder.
        """
        if pretrain:
            if x2 is None:
                raise ValueError("x2 must be provided for pre-training")

            # Online network forward pass
            online_feat1 = self.online_encoder(x1)
            online_feat2 = self.online_encoder(x2)
            online_proj1 = self.online_projector(online_feat1)
            online_proj2 = self.online_projector(online_feat2)
            pred1 = self.online_predictor(online_proj1)
            pred2 = self.online_predictor(online_proj2)

            # Target network forward pass (no gradients)
            with torch.no_grad():
                target_feat1 = self.target_encoder(x1)
                target_feat2 = self.target_encoder(x2)
                target_proj1 = self.target_projector(target_feat1).detach()
                target_proj2 = self.target_projector(target_feat2).detach()

            return pred1, pred2, target_proj1, target_proj2
        else:
            # Fine-tuning/Evaluation phase
            if self.classification_head is None:
                raise RuntimeError("Classification head not initialized. Call add_classification_head first.")
            # Use the online encoder features for classification
            features = self.online_encoder(x1)
            return self.classification_head(features)

    def add_classification_head(self, num_classes):
        """Adds the classification head for fine-tuning."""
        # Use the feature dimension from the online encoder
        self.classification_head = nn.Linear(self.feature_dim, num_classes)

    @staticmethod
    def regression_loss(x, y):
        """Calculates the normalized MSE loss between two vectors."""
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        # Calculate MSE loss and scale by 2 as per paper (sum over batch, mean over feature dim)
        loss = 2 - 2 * (x_norm * y_norm).sum(dim=-1)
        return loss.mean()

    def byol_loss(self, pred1, pred2, target_proj1, target_proj2):
        """Calculates the total BYOL loss (symmetric)."""
        loss1 = self.regression_loss(pred1, target_proj2)
        loss2 = self.regression_loss(pred2, target_proj1)
        return (loss1 + loss2) / 2 # Average the two losses

