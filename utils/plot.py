import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, precision_recall_curve

def plot_loss(results, save_file_path):
    plt.plot(results['train_loss'], label='Training Loss')
    plt.plot(results['test_loss'], label='Testing Loss')
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(save_file_path + 'loss.png'))
    plt.clf()

def plot_acc(results, save_file_path):
    # print(type(results['train_acc']))
    # print(type(results['test_acc']))
    plt.plot(torch.tensor(results['train_acc']).cpu(), label='Training Accuracy')
    plt.plot(torch.tensor(results['test_acc']).cpu(), label='Testing Accuracy')
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(str(save_file_path + 'accuracy.png'))
    plt.clf()

def plot_confusion_matrix(cm, positive_class_name: str, save_path: str):
    """
    Args:
        cm: Confusion matrix array (2x2)
        positive_class_name: Name of the positive class (e.g. "1_Amiloidose")
        save_path: Full path to save the plot (including filename)
    """
    class_names = ["Negative", positive_class_name]
    
    plt.figure(figsize=(10, 7))
    plt.title('Matriz de Confusão')
    plt.ylabel('Classes Verdadeiras')
    plt.xlabel('Classes Preditas')
    
    ax = sns.heatmap(cm, cmap="OrRd", 
                    annot=True, annot_kws={"size": 30}, 
                    fmt='g',
                    xticklabels=class_names,
                    yticklabels=class_names)
    
    # Rotate class names for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()  # Prevent label cutoff
    print(f"Salvando Matriz de Confusão em: {save_path}")
    plt.savefig(save_path)
    plt.clf()
    plt.close()