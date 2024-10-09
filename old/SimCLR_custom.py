import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import os
from PIL import Image
import numpy as np

# 1. Define the custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for sub_class in sorted(os.listdir(class_dir)):
                sub_class_dir = os.path.join(class_dir, sub_class)
                for img_name in sorted(os.listdir(sub_class_dir)):
                    img_path = os.path.join(sub_class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. Load and Preprocess the Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = CustomDataset(root_dir='./datasetMestradoGledson+gabriel', transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)

# 3. Data Augmentation Functions for SimCLR
class SimCLRTransform:
    def __init__(self, s=0.5):
        self.random_crop = transforms.RandomResizedCrop(size=224)  # Keep the size consistent with the rest of the pipeline
        self.color_distortion = transforms.Compose([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __call__(self, x):
        return self.random_crop(x), self.color_distortion(x)


simclr_transform = SimCLRTransform()

class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        x_i, x_j = simclr_transform(x)
        return x_i, x_j

    def __len__(self):
        return len(self.dataset)

simclr_train_set = SimCLRDataset(train_set)
simclr_train_loader = DataLoader(simclr_train_set, batch_size=256, shuffle=True, num_workers=2)

# 4. SimCLR Model Architecture
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

class ProjectionHead(nn.Module):
    def __init__(self, encoder):
        super(ProjectionHead, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

encoder = Encoder()
simclr_model = ProjectionHead(encoder)

# 5. Contrastive Loss Function
def contrastive_loss(z_i, z_j, temperature=0.1):
    batch_size = z_i.shape[0]
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    labels = torch.arange(batch_size)
    labels = torch.cat([labels, labels], dim=0).to(z_i.device)

    mask = torch.eye(batch_size * 2).to(z_i.device)
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)], dim=0)
    negatives = similarity_matrix[~mask.bool()].view(batch_size * 2, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# 6. Train the SimCLR Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simclr_model = simclr_model.to(device)
optimizer = optim.Adam(simclr_model.parameters(), lr=0.0003)

for epoch in range(10):
    simclr_model.train()
    running_loss = 0.0
    for i, (x_i, x_j) in enumerate(simclr_train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        optimizer.zero_grad()
        z_i = simclr_model(x_i)
        z_j = simclr_model(x_j)
        loss = contrastive_loss(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(simclr_train_loader)}')

# 7. Fine-tune and Evaluate on Labeled Data (Using F1 Score)
class FineTuneModel(nn.Module):
    def __init__(self, encoder):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, len(train_set.classes))  # Number of classes based on dataset

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

fine_tune_model = FineTuneModel(encoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.0003)

for epoch in range(10):
    fine_tune_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = fine_tune_model(images)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')

# 8. Predict and calculate F1 score
fine_tune_model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = fine_tune_model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f'SimCLR Test F1 Score: {f1}')

# 9. Baseline Model for Comparison (Using F1 Score)
baseline_model = Encoder()
baseline_model.fc = nn.Linear(512, len(train_set.classes))  # Number of classes based on dataset
baseline_model = baseline_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=0.0003)

for epoch in range(10):
    baseline_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = baseline_model(images)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')

# 10. Predict and calculate F1 score
baseline_model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = baseline_model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

baseline_f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f'Baseline Test F1 Score: {baseline_f1}')

# 11. Comparison of Results
print(f"SimCLR Test F1 Score: {f1}")
print(f"Baseline Test F1 Score: {baseline_f1}")

if f1 > baseline_f1:
    print("SimCLR outperformed the baseline model based on F1 Score.")
else:
    print("Baseline model outperformed SimCLR based on F1 Score.")
