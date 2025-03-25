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

def plot_confusion_matrix(cm, save_path):
    
    
    plt.figure(figsize = (10,7))
    plt.title('Matriz de Confusão')
    plt.ylabel('Classes Verdadeiras')
    plt.xlabel('Classes Preditas')
    sns.heatmap(cm,cmap="OrRd", annot=True, annot_kws={"size":30}, fmt='g')
    name=save_path + 'confusion_matrix.png'
    print(f"Saving Confusion Matrix in: {name}")
    plt.savefig(name) 
    plt.clf()