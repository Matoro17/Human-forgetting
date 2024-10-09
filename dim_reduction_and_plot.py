import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from custom_dataset import CustomDataset
from simclr_model import Encoder, ProjectionHead, FineTuneModel
from torchvision.models import ResNet18_Weights

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the test dataset
test_set = CustomDataset(root_dir='./datasetMestradoGledson+gabriel', transform=transform, split='test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

# Load class names
class_names = test_set.classes

# Load the trained SimCLR model
encoder = Encoder()
simclr_model = ProjectionHead(encoder)
simclr_model.load_state_dict(torch.load('simclr_model.pth'))
simclr_model = simclr_model.to(device)
simclr_model.eval()

# Load the fine-tuned model
fine_tune_model = FineTuneModel(encoder, num_classes=len(class_names))
fine_tune_model.load_state_dict(torch.load('fine_tune_model.pth'))
fine_tune_model = fine_tune_model.to(device)
fine_tune_model.eval()

# Extract feature vectors from the SimCLR model
simclr_features = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        features = simclr_model.encoder(images)  # Extract features from the encoder
        simclr_features.append(features.cpu().numpy())
        labels.extend(lbls.numpy())

simclr_features = np.concatenate(simclr_features, axis=0)
labels = np.array(labels)

# Dimensionality Reduction using t-SNE or PCA
# Option 1: t-SNE
tsne = TSNE(n_components=3, random_state=42)
reduced_features_tsne = tsne.fit_transform(simclr_features)

# Option 2: PCA
pca = PCA(n_components=3)
reduced_features_pca = pca.fit_transform(simclr_features)

# Plotting function with class names, interactive rotation, and fullscreen mode
def plot_3d(features, title, labels, class_names):
    # Create a figure and set it to fullscreen manually
    fig = plt.figure(figsize=(16, 9))  # Adjust size for your screen resolution
    ax = fig.add_subplot(111, projection='3d')
    
    # Enabling interactive mode
    plt.ion()
    
    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='viridis', s=20)
    
    # Add legend with class names
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    ax.add_artist(legend1)
    ax.set_title(title)
    
    # Map class labels to their respective names
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        ax.text(np.mean(features[labels == label, 0]),
                np.mean(features[labels == label, 1]),
                np.mean(features[labels == label, 2]),
                class_names[label],
                fontsize=12, weight="bold")

    plt.show()

    # Keep the plot open for interaction
    input("Press Enter to close the plot...")


# Plotting the SimCLR features
plot_3d(reduced_features_tsne, "SimCLR Features - t-SNE", labels, class_names)
plot_3d(reduced_features_pca, "SimCLR Features - PCA", labels, class_names)

# Extract feature vectors from the Fine-Tuned model
fine_tune_features = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        features = fine_tune_model.encoder(images)  # Extract features from the encoder
        fine_tune_features.append(features.cpu().numpy())

fine_tune_features = np.concatenate(fine_tune_features, axis=0)

# Dimensionality Reduction using t-SNE or PCA for Fine-Tuned model
reduced_fine_tune_tsne = tsne.fit_transform(fine_tune_features)
reduced_fine_tune_pca = pca.fit_transform(fine_tune_features)

# Plotting the Fine-Tuned model features
plot_3d(reduced_fine_tune_tsne, "Fine-Tuned Model Features - t-SNE", labels, class_names)
plot_3d(reduced_fine_tune_pca, "Fine-Tuned Model Features - PCA", labels, class_names)
