import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import copy
import math

class DINOHead(nn.Module):
    """DINO projection head.

    Based on the paper and user code, uses MLP with GELU and LayerNorm.
    Output is L2-normalized.
    Includes weight normalization trick from the paper.
    """
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # Original DINO uses 3 layers MLP with hidden_dim=2048, bottleneck_dim=256
        # The output dimension (out_dim) is typically large, e.g., 65536
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim), # Bottleneck layer
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1) # Initialize weight norm gain to 1
        self.last_layer.weight_g.requires_grad = False # Freeze gain

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # Normalize before last layer
        x = self.last_layer(x)
        return x

class DINO(nn.Module):
    """DINO model implementation.

    Uses student and teacher networks with the same architecture but different weights.
    Teacher weights are updated via EMA of student weights.
    """
    def __init__(self, backbone_name="resnet18", use_pretrained=True, out_dim=65536, projection_hidden_dim=2048, projection_bottleneck_dim=256):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_dim = out_dim

        # Student network
        self.student_backbone, self.feature_dim = self._create_backbone(backbone_name, use_pretrained)
        self.student_head = DINOHead(self.feature_dim, out_dim, projection_hidden_dim, projection_bottleneck_dim)

        # Teacher network (copied from student, frozen, updated with EMA)
        self.teacher_backbone, _ = self._create_backbone(backbone_name, use_pretrained=False) # Teacher often not pretrained
        self.teacher_head = DINOHead(self.feature_dim, out_dim, projection_hidden_dim, projection_bottleneck_dim)
        self._initial_sync()
        self._freeze_teacher()

        # Classification head (added during fine-tuning, uses student backbone)
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
        """Initializes teacher network with student network weights."""
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

    def _freeze_teacher(self):
        """Freezes the teacher network parameters."""
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher_network(self, momentum):
        """Performs EMA update of the teacher network."""
        for param_s, param_t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)

    def forward(self, views, pretrain=True, return_features=False):
        """Forward pass.

        Args:
            views (list or torch.Tensor): List of augmented views (for pretrain) or single tensor (for eval).
            pretrain (bool): If True, performs DINO pre-training forward pass.
            return_features (bool): If True and pretrain=False, returns backbone features instead of logits.
        """
        if pretrain:
            # Student forward pass for all views
            student_output = [self.student_head(self.student_backbone(view)) for view in views]

            # Teacher forward pass for global views only (first 2 views)
            with torch.no_grad():
                teacher_output = [self.teacher_head(self.teacher_backbone(view)) for view in views[:2]]

            return student_output, teacher_output
        else:
            # Fine-tuning/Evaluation phase (uses student backbone)
            if not isinstance(views, torch.Tensor):
                # Assume first view is the one for evaluation if multiple provided
                x = views[0]
            else:
                x = views

            features = self.student_backbone(x)
            if return_features:
                return features
            else:
                if self.classification_head is None:
                    raise RuntimeError("Classification head not initialized. Call add_classification_head first.")
                return self.classification_head(features)

    def add_classification_head(self, num_classes):
        """Adds the classification head for fine-tuning."""
        self.classification_head = nn.Linear(self.feature_dim, num_classes)

    @staticmethod
    def dino_loss(student_output, teacher_output, student_temp, teacher_temp, center):
        """Calculates the DINO loss.

        Args:
            student_output (list): List of student outputs (logits) for all views.
            teacher_output (list): List of teacher outputs (logits) for global views.
            student_temp (float): Student temperature.
            teacher_temp (float): Teacher temperature.
            center (torch.Tensor): Teacher center.
        """
        s_global1, s_global2 = student_output[0], student_output[1]
        t_global1, t_global2 = teacher_output[0], teacher_output[1]

        # Apply temperature and softmax to teacher outputs (centered)
        t_softmax1 = F.softmax((t_global1 - center) / teacher_temp, dim=-1).detach()
        t_softmax2 = F.softmax((t_global2 - center) / teacher_temp, dim=-1).detach()

        # Calculate cross-entropy loss for each student view against teacher views
        total_loss = 0
        n_loss_terms = 0

        for i, s_out in enumerate(student_output):
            s_softmax = F.log_softmax(s_out / student_temp, dim=-1)
            # Each student view tries to match both teacher global views
            if i < 2: # Global student views match the other global teacher view
                loss1 = - (t_softmax2 * s_softmax).sum(dim=-1)
                loss2 = - (t_softmax1 * s_softmax).sum(dim=-1)
                total_loss += (loss1.mean() + loss2.mean()) / 2 # Average loss for the two global views
                n_loss_terms +=1
            else: # Local student views match both global teacher views
                loss1 = - (t_softmax1 * s_softmax).sum(dim=-1)
                loss2 = - (t_softmax2 * s_softmax).sum(dim=-1)
                total_loss += (loss1.mean() + loss2.mean()) / 2 # Average loss for local views
                n_loss_terms +=1

        # Original paper averages loss over all pairs (student_view, teacher_global_view)
        # Here we average the loss contribution of each student view
        return total_loss / n_loss_terms

