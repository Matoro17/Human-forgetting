import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from training.train_simclr import train_simclr
from training.train_byol import train_byol
from training.train_dino import train_dino
from training.fine_tune import fine_tune
from utils.evaluation import evaluate
from models.encoder import Encoder

from custom_dataset import CustomDataset

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CustomDataset(root_dir=DATASET_DIR).classes)
    
    # 1. Train SimCLR
    simclr_encoder = train_simclr(device)
    simclr_fine_tuned_model = fine_tune(simclr_encoder, device, num_classes)
    simclr_f1 = evaluate(simclr_fine_tuned_model, device, num_classes)
    print(f'SimCLR F1 Score: {simclr_f1}')
    
    # 2. Train BYOL
    byol_encoder = train_byol(device)
    byol_fine_tuned_model = fine_tune(byol_encoder, device, num_classes)
    byol_f1 = evaluate(byol_fine_tuned_model, device, num_classes)
    print(f'BYOL F1 Score: {byol_f1}')
    
    # # 3. Train DINO
    # dino_encoder = train_dino(device)
    # dino_fine_tuned_model = fine_tune(dino_encoder, device, num_classes)
    # dino_f1 = evaluate(dino_fine_tuned_model, device, num_classes)
    # print(f'DINO F1 Score: {dino_f1}')
    
    # 4. Train Baseline (no SSL)
    baseline_encoder = Encoder().to(device)
    baseline_fine_tuned_model = fine_tune(baseline_encoder, device, num_classes)
    baseline_f1 = evaluate(baseline_fine_tuned_model, device, num_classes)
    print(f'Baseline F1 Score: {baseline_f1}')
    
    # 5. Print Results
    print("\nComparison of F1 Scores:")
    print(f"SimCLR: {simclr_f1}")
    print(f"BYOL: {byol_f1}")
    # print(f"DINO: {dino_f1}")
    print(f"Baseline: {baseline_f1}")

if __name__ == "__main__":
    main()
