import torch
import torch.nn as nn
import copy

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DINOHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class DINO(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(DINO, self).__init__()
        # Student network
        self.student_encoder = encoder
        self.student_projection = DINOHead(512, projection_dim)

        # Teacher network (initialized as a copy of the student network)
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_projection = copy.deepcopy(self.student_projection)

        # Freeze the teacher network
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_projection.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        # Student network forward pass
        student_proj1 = self.student_projection(self.student_encoder(x1))
        student_proj2 = self.student_projection(self.student_encoder(x2))

        # Teacher network forward pass (no gradients)
        with torch.no_grad():
            teacher_proj1 = self.teacher_projection(self.teacher_encoder(x1))
            teacher_proj2 = self.teacher_projection(self.teacher_encoder(x2))

        return student_proj1, student_proj2, teacher_proj1.detach(), teacher_proj2.detach()
