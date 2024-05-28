import torch
import torchvision.transforms.functional as F
import random

def rotate_image(image, rotation):
    if rotation == 0:
        return image
    elif rotation == 90:
        return F.rotate(image, 90)
    elif rotation == 180:
        return F.rotate(image, 180)
    elif rotation == 270:
        return F.rotate(image, 270)

def generate_rotation_batch(batch):
    images, labels = batch
    rotated_images = []
    rotation_labels = []
    for img in images:
        rotation = random.choice([0, 90, 180, 270])
        rotated_img = rotate_image(img, rotation)
        rotated_images.append(rotated_img)
        rotation_labels.append(rotation // 90)
    return torch.stack(rotated_images), torch.tensor(rotation_labels)