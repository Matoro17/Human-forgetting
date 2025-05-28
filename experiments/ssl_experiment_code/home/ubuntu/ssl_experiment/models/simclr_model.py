import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

class SimCLR(nn.Module):
    """SimCLR model adapted from user code and review.

    Uses a backbone (ResNet) and a projection head.
    Includes methods for forward pass and NT-Xent loss calculation.
    """
    def __init__(self, backbone_name="resnet18", use_pretrained=True, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.backbone, self.feature_dim = self._create_backbone(backbone_name, use_pretrained)

        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), # Hidden layer size from original code
            nn.BatchNorm1d(512), # Added BatchNorm based on common practices
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

        # Remove the original classifier (fc layer) from the backbone
        self.backbone.fc = nn.Identity()

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
        return backbone, feature_dim

    def forward(self, x, pretrain=True):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            pretrain (bool): If True, returns backbone features and projections (for pre-training).
                             If False, returns classification logits (for fine-tuning/evaluation).
        """
        h = self.backbone(x) # Backbone features

        if pretrain:
            z = self.projection_head(h) # Projections
            # Return normalized features and projections as per SimCLR paper
            return F.normalize(h, dim=-1), F.normalize(z, dim=-1)
        else:
            # Fine-tuning/Evaluation phase
            if self.classification_head is None:
                raise RuntimeError("Classification head not initialized. Call add_classification_head first.")
            return self.classification_head(h) # Logits

    def add_classification_head(self, num_classes):
        """Adds the classification head for fine-tuning."""
        self.classification_head = nn.Linear(self.feature_dim, num_classes)

    def nt_xent_loss(self, z_i, z_j, temperature=0.5):
        """Calculates the Normalized Temperature-scaled Cross Entropy loss."""
        batch_size = z_i.shape[0]
        device = z_i.device

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)

        # Calculate cosine similarity matrix
        # representations shape: [2*B, D]
        # similarity_matrix shape: [2*B, 2*B]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

        # Mask to identify positive pairs (diagonal elements after concatenation)
        # Create a mask for positive pairs (i, i+B) and (i+B, i)
        mask = torch.eye(batch_size, device=device).repeat(2, 2)
        mask = mask[~torch.eye(2 * batch_size, dtype=torch.bool, device=device)].view(2 * batch_size, -1)

        # Select positive samples (similarity between z_i and z_j)
        # Positive pairs are at (i, i+B) and (i+B, i)
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ]).unsqueeze(1) # Shape: [2*B, 1]

        # Select negative samples (all pairs except self-comparisons and positive pairs)
        negatives = similarity_matrix[~torch.eye(2 * batch_size, dtype=torch.bool, device=device)].view(2 * batch_size, -1)
        # negatives shape: [2*B, 2*B-2]

        # Combine positive and negative samples for logits
        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        # Labels: positive pair is always the first element (index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        return loss

