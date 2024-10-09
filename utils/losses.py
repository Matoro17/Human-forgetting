import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(z_i, z_j, temperature=0.1):
    batch_size = z_i.shape[0]
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    labels = torch.arange(batch_size).to(z_i.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(batch_size * 2).to(z_i.device)
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)], dim=0)
    negatives = similarity_matrix[~mask.bool()].view(batch_size * 2, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss



def byol_loss(online_proj1, online_proj2, target_proj1, target_proj2):
    """
    Calculate the BYOL loss as the mean squared error between the online network
    projections and the target network projections.
    """
    # Normalize projections to have unit norm
    online_proj1 = nn.functional.normalize(online_proj1, dim=1)
    online_proj2 = nn.functional.normalize(online_proj2, dim=1)
    target_proj1 = nn.functional.normalize(target_proj1, dim=1)
    target_proj2 = nn.functional.normalize(target_proj2, dim=1)
    
    # Calculate the mean squared error loss between online and target projections
    loss1 = torch.mean((online_proj1 - target_proj2) ** 2)
    loss2 = torch.mean((online_proj2 - target_proj1) ** 2)
    
    return 0.5 * (loss1 + loss2)


def dino_loss(student_proj1, student_proj2, teacher_proj1, teacher_proj2, temperature=0.07):
    """
    Calculate the DINO loss as the cross-entropy between the student outputs
    and the teacher's soft labels.
    """
    # Normalize projections
    student_proj1 = F.normalize(student_proj1, dim=1)
    student_proj2 = F.normalize(student_proj2, dim=1)
    teacher_proj1 = F.normalize(teacher_proj1, dim=1)
    teacher_proj2 = F.normalize(teacher_proj2, dim=1)
    
    # Softmax with temperature for teacher and student
    student_logits1 = student_proj1 / temperature
    student_logits2 = student_proj2 / temperature
    teacher_probs1 = F.softmax(teacher_proj1 / temperature, dim=1)
    teacher_probs2 = F.softmax(teacher_proj2 / temperature, dim=1)
    
    # Cross-entropy loss
    loss1 = -torch.mean(torch.sum(teacher_probs2 * F.log_softmax(student_logits1, dim=1), dim=1))
    loss2 = -torch.mean(torch.sum(teacher_probs1 * F.log_softmax(student_logits2, dim=1), dim=1))
    
    return 0.5 * (loss1 + loss2)