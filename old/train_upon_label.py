import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import pytorch_lightning as pl
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD
from multiprocessing import cpu_count

# Utility functions

# Default value setter
def default(val, def_val):
    return def_val if val is None else val

# Set random seeds for reproducibility
def reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Move tensor t1 to the same device as tensor t2
def device_as(t1, t2):
    return t1.to(t2.device)

# Update model weights from a checkpoint
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model

# Data augmentation class
class Augment:
    def __init__(self, img_size, s=1):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([blur], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

# Contrastive loss function for SimCLR
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, a, b):
        # Concatenate representations and calculate cosine similarity
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.shape[0]
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

# Projection head added to the backbone
class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = config.embedding_size
        self.backbone = default(model, models.resnet18(pretrained=False, num_classes=config.embedding_size))
        mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
        print('Dim MLP input:', mlp_dim)
        self.backbone.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)

# Define parameter groups for optimizer
def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups

# PyTorch Lightning module for SimCLR
class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config
        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)
        self.loss = ContrastiveLoss(temperature=self.config.temperature)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, weight_decay=self.config.weight_decay, optimizer_name=self.config.optimizer)
        
        optimizer = SGD(param_groups, lr=self.config.lr, momentum=self.config.momentum)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.config.warmup_epochs, max_epochs=max_epochs
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        augment = Augment(img_size=32)
        dataset = CustomImageDataset(root_dir=self.config.data_dir, transform=augment)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=cpu_count())
        return dataloader

# Custom dataset for image data
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._gather_images(self.root_dir)
        print(f"Collected {len(self.image_paths)} images.")

    def _gather_images(self, root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        print(f"Example image paths: {image_paths[:5]}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            img1, img2 = self.transform(image)
            return img1, img2
        else:
            return image, image

# Labeled dataset for classification
class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._gather_images_and_labels(self.root_dir)
        print(f"Collected {len(self.image_paths)} images with labels.")

    def _gather_images_and_labels(self, root_dir):
        image_paths = []
        labels = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(os.path.basename(root))
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_to_idx[label] for label in labels]
        print(f"Example image paths: {image_paths[:5]}")
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Classification model for fine-tuning
class ClassificationModel(pl.LightningModule):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = config
        self.model = model if model is not None else models.resnet18(pretrained=False, num_classes=config.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = LabeledImageDataset(root_dir=self.config.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=cpu_count())
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = LabeledImageDataset(root_dir=self.config.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=cpu_count())
        return dataloader

# Main execution
if __name__ == "__main__":
    # Define the configuration
    config_ssl = Namespace(
        data_dir='datasetMestradoGledson+gabriel/0_Amiloidose/AZAN',
        batch_size=128,
        epochs=100,
        warmup_epochs=10,
        temperature=0.5,
        lr=0.3,
        weight_decay=1e-4,
        momentum=0.9,
        optimizer='sgd',
        embedding_size=128
    )

    config_finetune = Namespace(
        data_dir='datasetMestradoGledson+gabriel',
        batch_size=32,
        epochs=50,
        lr=0.001,
        num_classes=5  # Adjust based on your dataset
    )

    # Set random seed for reproducibility
    seed = 42
    reproducibility(seed)

    # Initialize and train the SimCLR model (self-supervised learning)
    simclr_model = SimCLR_pl(config_ssl)
    trainer_ssl = pl.Trainer(
        max_epochs=config_ssl.epochs,
        accelerator='gpu', devices=1 if torch.cuda.is_available() else 0,
        enable_progress_bar=True
    )
    trainer_ssl.fit(simclr_model)

    # Initialize and train the classification model (fine-tuning)
    backbone_model = simclr_model.model.backbone
    classification_model = ClassificationModel(config_finetune, model=backbone_model)
    trainer_finetune = pl.Trainer(
        max_epochs=config_finetune.epochs,
        accelerator='gpu', devices=1 if torch.cuda.is_available() else 0,
        enable_progress_bar=True
    )
    trainer_finetune.fit(classification_model)
