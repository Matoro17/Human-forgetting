import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from torchvision import transforms
import os
from dotenv import load_dotenv

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

    # Get the class names
    class_names = test_set.classes

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

    # Generate a classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    # Extract F1 scores for each class
    f1_per_class = {class_names[i]: report[class_names[i]]['f1-score'] for i in range(num_classes)}
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    # Optionally, save these metrics to a file
    save_metrics_to_csv(f1_per_class, 'per_class_f1_scores.csv')
    
    return f1_per_class, macro_f1, weighted_f1


import csv

def save_metrics_to_csv(metrics, filepath):
    with open(filepath, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'F1 Score'])
        for metric, value in metrics.items():
            writer.writerow([metric, value])
