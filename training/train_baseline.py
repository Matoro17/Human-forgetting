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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CustomDataset(root_dir=DATASET_DIR).classes)
# 4. Train Baseline (no SSL)
baseline_encoder = Encoder().to(device)
baseline_fine_tuned_model = fine_tune(baseline_encoder, device, num_classes)
baseline_f1 = evaluate(baseline_fine_tuned_model, device, num_classes)
print(f'Baseline F1 Score: {baseline_f1}')