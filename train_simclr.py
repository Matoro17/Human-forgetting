import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from simclr_model import Encoder, ProjectionHead
from custom_dataset import CustomDataset

# Data Augmentation for SimCLR
class SimCLRTransform:
    def __init__(self, s=0.5):
        self.random_crop = transforms.RandomResizedCrop(size=224)
        self.color_distortion = transforms.Compose([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3)
        ])

    def __call__(self, x):
        return self.random_crop(x), self.color_distortion(x)

class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        x_i, x_j = SimCLRTransform()(x)
        return x_i, x_j

    def __len__(self):
        return len(self.dataset)

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

class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Load and Preprocess Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Initialize datasets
    train_set = CustomDataset(root_dir='./datasetMestradoGledson', transform=transform, split='train')
    test_set = CustomDataset(root_dir='./datasetMestradoGledson', transform=transform, split='test')

    # Initialize data loaders
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    simclr_train_set = SimCLRDataset(train_set)
    simclr_train_loader = DataLoader(simclr_train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # Initialize the SimCLR model
    encoder = Encoder()
    simclr_model = ProjectionHead(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_model = simclr_model.to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.0003)

    # Train SimCLR Model
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

    # Save the trained SimCLR model
    torch.save(simclr_model.state_dict(), 'simclr_model.pth')
    print("SimCLR model saved as simclr_model.pth")

    # Fine-Tune the Model
    fine_tune_model = FineTuneModel(encoder, num_classes=len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.0003)

    for epoch in range(10):
        fine_tune_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = fine_tune_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')

    # Save the fine-tuned model
    torch.save(fine_tune_model.state_dict(), 'fine_tune_model.pth')
    print("Fine-tuned model saved as fine_tune_model.pth")

    # Predict and calculate F1 score on the test set
    fine_tune_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = fine_tune_model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'SimCLR Test F1 Score: {f1}')

    # Train Baseline Model for Comparison
    baseline_model = Encoder()
    baseline_model.fc = nn.Linear(512, len(train_set.classes))
    baseline_model = baseline_model.to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.0003)

    for epoch in range(10):
        baseline_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')

    # Evaluate Baseline Model
    baseline_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = baseline_model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    baseline_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Baseline Test F1 Score: {baseline_f1}')

    # Comparison of Results
    print(f"SimCLR Test F1 Score: {f1}")
    print(f"Baseline Test F1 Score: {baseline_f1}")

    if f1 > baseline_f1:
        print("SimCLR outperformed the baseline model based on F1 Score.")
    else:
        print("Baseline model outperformed SimCLR based on F1 Score.")
