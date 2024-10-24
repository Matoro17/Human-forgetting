import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from torchvision import transforms


import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def evaluate(model, device, num_classes):
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Initialize the test set with the transform
    test_set = CustomDataset(root_dir=DATASET_DIR, split='test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return f1

import csv

def save_metrics_to_csv(metrics, filepath):
    with open(filepath, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for metric, value in metrics.items():
            writer.writerow([metric, value])

