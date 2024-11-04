import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class DINO(nn.Module):
    def __init__(self, student_arch='resnet50', teacher_arch='resnet50', use_pretrained=True, out_dim=256, momentum=0.996):
        super(DINO, self).__init__()
        
        # Initialize student and teacher networks
        self.student = self._create_network(student_arch, use_pretrained, out_dim)
        self.teacher = self._create_network(teacher_arch, use_pretrained, out_dim)
        
        # Initialize teacher parameters with student parameters
        self._initialize_teacher()

        # Momentum for teacher update
        self.momentum = momentum

    def _create_network(self, arch, use_pretrained, out_dim):
        if arch == 'resnet50':
            # Load the backbone without the fully connected layer
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            backbone.fc = nn.Identity()  # Remove the original fully connected layer
            
            # Determine the output feature size
            feature_dim = self._get_backbone_output_dim(backbone)
            
            # Projection head for the student/teacher network
            projection_head = nn.Sequential(
                nn.Linear(feature_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_dim)
            )
            return nn.Sequential(backbone, projection_head)
        else:
            raise ValueError("Currently, only resnet50 is supported.")

    def _get_backbone_output_dim(self, backbone):
        # Run a dummy input to determine the output feature size
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = backbone(dummy_input)
        return features.shape[1]

    def _initialize_teacher(self):
        # Copy student parameters to teacher
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False  # Freeze the teacher parameters

    def update_teacher(self):
        # Update the teacher parameters with momentum
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1.0 - self.momentum) * student_param.data

    def forward(self, x):
        # Forward pass through the student network
        student_output = self.student(x)
        
        # No gradient for the teacher
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        # Return both student and teacher outputs
        return student_output, teacher_output

    def loss(self, student_output, teacher_output):
        # Apply DINO loss (for simplicity, using cosine similarity)
        student_output = F.normalize(student_output, dim=-1)
        teacher_output = F.normalize(teacher_output, dim=-1)
        loss = 2 - 2 * (student_output * teacher_output).sum(dim=-1).mean()
        return loss
