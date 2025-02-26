import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv
import numpy as np

def evaluate(log_filepath, model, device, test_loader, num_classes, class_names, save_csv=True):
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

    # Determine the unique classes in the data
    unique_classes = np.unique(all_labels + all_predictions)
    if len(unique_classes) != num_classes:
        log_message(log_filepath, f"Warning: Number of unique classes in data ({len(unique_classes)}) does not match num_classes ({num_classes}).")
        log_message(log_filepath, f"Unique classes in data: {unique_classes}")
        log_message(log_filepath, f"Expected classes: {class_names}")

    # Generate a classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        labels=np.arange(len(class_names)),  # Ensure labels match class_names
        output_dict=True
    )
    
    # Extract F1 scores
    f1_per_class = {class_names[i]: report[class_names[i]]['f1-score'] for i in range(len(class_names))}
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    if save_csv:
        save_metrics_to_txt(f1_per_class, 'per_class_f1_scores.csv')

    return f1_per_class, macro_f1, weighted_f1

def save_metrics_to_txt(results, filepath):
    if not results:
        print("No results to save.")
        return

    with open(filepath, mode='w') as file:
        for arch, metrics in results.items():
            file.write(f"Architecture: {arch}\n")
            for key, value in metrics.items():
                file.write(f"  {key}: {value}\n")
            file.write("\n")  # Separate architectures with a blank line


def log_message(log_filepath, message):
    """Logs a message to both the console and a file."""
    print(message)
    with open(log_filepath, "a") as log_file:
        log_file.write(message + "\n")
