from torchvision import transforms
from PIL import Image
import torch # Added for ToTensor check

# SimCLR Augmentation (as per Appendix A in the paper)
def get_simclr_transform(image_size=224):
    s = 1.0
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    gaussian_blur_p = 0.5
    kernel_size = int(0.1 * image_size)
    if kernel_size % 2 == 0: kernel_size += 1

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=gaussian_blur_p),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class SimCLRTransform:
    """Applies SimCLR transform twice to get two correlated views."""
    def __init__(self, image_size=224):
        self.transform = get_simclr_transform(image_size)

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

# BYOL Augmentation (as per Appendix F in the paper)
def get_byol_transform(image_size=224):
    s = 1.0
    color_jitter_params = (0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    kernel_size = int(0.1 * image_size)
    if kernel_size % 2 == 0: kernel_size += 1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # View 1 Transform (includes blur)
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter_params)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        normalize
    ])

    # View 2 Transform (excludes blur, includes solarization)
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter_params)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.2), # Corrected threshold to int
        transforms.ToTensor(),
        normalize
    ])
    return transform1, transform2

class BYOLTransform:
    """Applies BYOL transforms to get two asymmetric views."""
    def __init__(self, image_size=224):
        self.transform1, self.transform2 = get_byol_transform(image_size)

    def __call__(self, x):
        view1 = self.transform1(x)
        view2 = self.transform2(x)
        return view1, view2

# DINO Augmentation (Multi-Crop)
class DINOTransform:
    """Implements DINO's multi-crop augmentation.

    Generates 2 global views and a specified number of local views.
    Corrected order of ToTensor and Normalize.
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), num_local_crops=6, image_size=224, local_image_size=96):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Color jittering parameters
        color_jitter_params = dict(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

        # Gaussian blur parameters
        kernel_size_global = int(0.1 * image_size)
        if kernel_size_global % 2 == 0: kernel_size_global += 1
        kernel_size_local = int(0.1 * local_image_size)
        if kernel_size_local % 2 == 0: kernel_size_local += 1

        gaussian_blur_global = transforms.GaussianBlur(kernel_size=kernel_size_global, sigma=(0.1, 2.0))
        gaussian_blur_local = transforms.GaussianBlur(kernel_size=kernel_size_local, sigma=(0.1, 2.0))

        # --- Global View 1 Transform (Flattened Compose) ---
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(**color_jitter_params)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([gaussian_blur_global], p=1.0),
            transforms.ToTensor(), # ToTensor before Normalize
            normalize,
        ])

        # --- Global View 2 Transform (Flattened Compose) ---
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(**color_jitter_params)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([gaussian_blur_global], p=0.1),
            transforms.RandomSolarize(128, p=0.2),
            transforms.ToTensor(), # ToTensor before Normalize
            normalize,
        ])

        # --- Local Views Transform (Flattened Compose) ---
        self.num_local_crops = num_local_crops
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_image_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(**color_jitter_params)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([gaussian_blur_local], p=0.5),
            transforms.ToTensor(), # ToTensor before Normalize
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))
        return crops

# Standard Transform for Fine-tuning/Evaluation
def get_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

