import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import csv
from .logging_utils import log_message

def calculate_metrics(y_true, y_pred, num_classes, class_names=None):
    """Calculates accuracy, F1 macro, and F1 per class."""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Calculate F1 per class, ensuring labels cover 0 to num_classes-1
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist() # Convert numpy array to list for easier saving
    }

    # Add F1 per class with meaningful names if provided
    if class_names and len(class_names) == num_classes:
        for i, class_name in enumerate(class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    else:
        for i in range(num_classes):
            metrics[f'f1_class_{i}'] = f1_per_class[i]

    return metrics

def save_fold_results(results_dict, fold, filename, log_filepath):
    """Appends results of a single fold to a CSV file."""
    file_exists = os.path.isfile(filename)
    # Flatten the dictionary for CSV writing
    flat_results = {'fold': fold}
    for model_name, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            # Handle confusion matrix separately if needed, or just save scalar metrics
            if not isinstance(value, list): # Exclude confusion matrix from simple CSV row
                 flat_results[f'{model_name}_{metric_name}'] = value

    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=flat_results.keys())
            if not file_exists:
                writer.writeheader() # Write header only if file is new
            writer.writerow(flat_results)
        log_message(log_filepath, f"Fold {fold} results appended to {filename}")
    except Exception as e:
        log_message(log_filepath, f"Error saving fold {fold} results to {filename}: {e}")

def save_aggregated_results(aggregated_results, filename, log_filepath):
    """Saves aggregated results (mean and std dev) across folds to a text file."""
    try:
        with open(filename, "w") as f:
            f.write("Aggregated Results (Mean ± Std Dev across Folds):\n")
            f.write("==================================================\n")
            for model_name, metrics in aggregated_results.items():
                f.write(f"\nModel: {model_name}\n")
                f.write("--------------------\n")
                for metric_name, values in metrics.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}\n")
        log_message(log_filepath, f"Aggregated results saved to {filename}")
    except Exception as e:
        log_message(log_filepath, f"Error saving aggregated results to {filename}: {e}")


