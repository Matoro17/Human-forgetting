import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv

def evaluate(model, device, test_loader, num_classes, class_names, save_csv=True):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Ensure outputs is a tensor, not a tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Generate a classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    # Extract F1 scores
    f1_per_class = {class_names[i]: report[class_names[i]]['f1-score'] for i in range(num_classes)}
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    if save_csv:
        save_metrics_to_csv(f1_per_class, 'per_class_f1_scores.csv')

    return f1_per_class, macro_f1, weighted_f1

def save_metrics_to_csv(metrics, filepath):
    with open(filepath, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'F1 Score'])
        for class_name, f1_score in metrics.items():
            writer.writerow([class_name, f1_score])
