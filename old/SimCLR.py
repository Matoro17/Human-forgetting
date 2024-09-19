import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import numpy as np

# 1. Load and Preprocess the Dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std dev of 0.5 for each channel.
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # Load the CIFAR-10 training set.
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  # Load the CIFAR-10 test set.

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)  # Create a data loader for the training set.
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)  # Create a data loader for the test set.

# 2. Data Augmentation Functions
class SimCLRTransform:
    def __init__(self, s=0.5):
        # Initialize data augmentation with random cropping and color distortion.
        self.random_crop = transforms.RandomResizedCrop(size=32)
        self.color_distortion = transforms.Compose([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s),  # Apply color jitter with certain strength.
            transforms.RandomGrayscale(p=0.2)  # Randomly convert images to grayscale with probability 0.2.
        ])

    def __call__(self, x):
        # Apply the transformations and return two augmented versions of the same image.
        return self.random_crop(x), self.color_distortion(x)

simclr_transform = SimCLRTransform()  # Instantiate the SimCLR transformation.

class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # Initialize with the original dataset.

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # Get an image from the dataset.
        x_i, x_j = simclr_transform(x)  # Apply the SimCLR transformation to create two views.
        return x_i, x_j  # Return the two transformed images.

    def __len__(self):
        return len(self.dataset)  # Return the size of the dataset.

simclr_train_set = SimCLRDataset(train_set)  # Create a new dataset with SimCLR augmentation.
simclr_train_loader = DataLoader(simclr_train_set, batch_size=256, shuffle=True, num_workers=2)  # Data loader for the augmented dataset.

# 3. SimCLR Model Architecture
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)  # Use ResNet-18 as the encoder backbone.
        self.backbone.fc = nn.Identity()  # Replace the fully connected layer with an identity layer (no-op).

    def forward(self, x):
        return self.backbone(x)  # Forward pass through the backbone.

class ProjectionHead(nn.Module):
    def __init__(self, encoder):
        super(ProjectionHead, self).__init__()
        self.encoder = encoder  # Use the encoder defined earlier.
        self.fc1 = nn.Linear(512, 128)  # First fully connected layer to reduce dimensions.
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer for further dimension reduction.

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder.
        x = F.relu(self.fc1(x))  # Apply ReLU after the first FC layer.
        x = self.fc2(x)  # Pass through the second FC layer.
        return x  # Return the final projection.

encoder = Encoder()  # Instantiate the encoder.
simclr_model = ProjectionHead(encoder)  # Instantiate the SimCLR model by adding the projection head to the encoder.

# 4. Contrastive Loss Function
def contrastive_loss(z_i, z_j, temperature=0.1):
    batch_size = z_i.shape[0]
    z_i = F.normalize(z_i, dim=1)  # Normalize the projections.
    z_j = F.normalize(z_j, dim=1)  # Normalize the projections.

    representations = torch.cat([z_i, z_j], dim=0)  # Concatenate positive pairs.
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # Compute cosine similarity.

    labels = torch.arange(batch_size)  # Create labels for positive pairs.
    labels = torch.cat([labels, labels], dim=0).to(z_i.device)  # Double the labels for the concatenated batch.
    
    mask = torch.eye(batch_size * 2).to(z_i.device)  # Mask to exclude self-similarity.
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)], dim=0)  # Extract positive similarities.
    negatives = similarity_matrix[~mask.bool()].view(batch_size * 2, -1)  # Extract negative similarities.

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)  # Create logits for the softmax.
    logits /= temperature  # Scale logits by temperature.

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)  # Labels for cross-entropy (positives are 0).
    loss = F.cross_entropy(logits, labels)  # Compute cross-entropy loss.
    return loss

# 5. Train the SimCLR Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU.
simclr_model = simclr_model.to(device)  # Move the model to the device.
optimizer = optim.Adam(simclr_model.parameters(), lr=0.0003)  # Set up the optimizer.

for epoch in range(10):  # Train for 10 epochs.
    simclr_model.train()  # Set the model to training mode.
    running_loss = 0.0  # Initialize running loss.
    for i, (x_i, x_j) in enumerate(simclr_train_loader):  # Iterate over the SimCLR training data.
        x_i, x_j = x_i.to(device), x_j.to(device)  # Move data to the device.
        
        optimizer.zero_grad()  # Clear gradients.
        z_i = simclr_model(x_i)  # Forward pass for the first view.
        z_j = simclr_model(x_j)  # Forward pass for the second view.
        loss = contrastive_loss(z_i, z_j)  # Compute the contrastive loss.
        loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update the model parameters.
        
        running_loss += loss.item()  # Accumulate the loss.
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(simclr_train_loader)}')  # Print the average loss for the epoch.

# 6. Fine-tune and Evaluate on Labeled Data (Using F1 Score)
class FineTuneModel(nn.Module):
    def __init__(self, encoder):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder  # Use the pre-trained encoder.
        self.fc = nn.Linear(512, 10)  # Fully connected layer for classification (10 classes in CIFAR-10).

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder.
        x = self.fc(x)  # Classify the output.
        return x

fine_tune_model = FineTuneModel(encoder).to(device)  # Instantiate and move the fine-tuning model to the device.
criterion = nn.CrossEntropyLoss()  # Set the loss function.
optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.0003)  # Set up the optimizer.

for epoch in range(10):  # Fine-tune for 10 epochs.
    fine_tune_model.train()  # Set the model to training mode.
    running_loss = 0.0  # Initialize running loss.
    for images, labels in train_loader:  # Iterate over the labeled training data.
        images, labels = images.to(device), labels.to(device)  # Move data to the device.
        
        optimizer.zero_grad()  # Clear gradients.
        outputs = fine_tune_model(images)  # Forward pass through the fine-tuning model.
        loss = criterion(outputs, labels.squeeze())  # Compute the cross-entropy loss.
        loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update the model parameters.
        
        running_loss += loss.item()  # Accumulate the loss.
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')  # Print the average loss for the epoch.

# Predict and calculate F1 score
fine_tune_model.eval()  # Set the model to evaluation mode.
all_labels = []
all_predictions = []

with torch.no_grad():  # Disable gradient calculation for evaluation.
    for images, labels in test_loader:  # Iterate over the test data.
        images = images.to(device)  # Move images to the device.
        outputs = fine_tune_model(images)  # Forward pass through the model.
        _, predicted = torch.max(outputs, 1)  # Get the predicted labels.
        all_labels.extend(labels.cpu().numpy())  # Store the true labels.
        all_predictions.extend(predicted.cpu().numpy())  # Store the predicted labels.

f1 = f1_score(all_labels, all_predictions, average='weighted')  # Calculate the weighted F1 score.
print(f'SimCLR Test F1 Score: {f1}')  # Print the F1 score for the SimCLR model.

# 7. Baseline Model for Comparison (Using F1 Score)
baseline_model = Encoder()  # Instantiate the baseline model using the encoder.
baseline_model.fc = nn.Linear(512, 10)  # Add a classification head for CIFAR-10.
baseline_model = baseline_model.to(device)  # Move the baseline model to the device.
criterion = nn.CrossEntropyLoss()  # Set the loss function.
optimizer = optim.Adam(baseline_model.parameters(), lr=0.0003)  # Set up the optimizer.

for epoch in range(10):  # Train the baseline model for 10 epochs.
    baseline_model.train()  # Set the model to training mode.
    running_loss = 0.0  # Initialize running loss.
    for images, labels in train_loader:  # Iterate over the labeled training data.
        images, labels = images.to(device), labels.to(device)  # Move data to the device.
        
        optimizer.zero_grad()  # Clear gradients.
        outputs = baseline_model(images)  # Forward pass through the baseline model.
        loss = criterion(outputs, labels.squeeze())  # Compute the cross-entropy loss.
        loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update the model parameters.
        
        running_loss += loss.item()  # Accumulate the loss.
    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader)}')  # Print the average loss for the epoch.

# Predict and calculate F1 score
baseline_model.eval()  # Set the model to evaluation mode.
all_labels = []
all_predictions = []

with torch.no_grad():  # Disable gradient calculation for evaluation.
    for images, labels in test_loader:  # Iterate over the test data.
        images = images.to(device)  # Move images to the device.
        outputs = baseline_model(images)  # Forward pass through the model.
        _, predicted = torch.max(outputs, 1)  # Get the predicted labels.
        all_labels.extend(labels.cpu().numpy())  # Store the true labels.
        all_predictions.extend(predicted.cpu().numpy())  # Store the predicted labels.

baseline_f1 = f1_score(all_labels, all_predictions, average='weighted')  # Calculate the weighted F1 score for the baseline model.
print(f'Baseline Test F1 Score: {baseline_f1}')  # Print the F1 score for the baseline model.

# 8. Comparison of Results
print(f"SimCLR Test F1 Score: {f1}")  # Print the SimCLR F1 score.
print(f"Baseline Test F1 Score: {baseline_f1}")  # Print the baseline F1 score.

# Compare the F1 scores and print the better-performing model.
if f1 > baseline_f1:
    print("SimCLR outperformed the baseline model based on F1 Score.")
else:
    print("Baseline model outperformed SimCLR based on F1 Score.")