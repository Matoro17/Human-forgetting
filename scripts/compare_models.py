import sys
import os
import torch

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from training.train_simclr import train_simclr
from training.train_byol import train_byol
from training.train_dino import train_dino
from training.fine_tune import fine_tune
from utils.evaluation import evaluate
from models.encoder import Encoder
from custom_dataset import CustomDataset
from dotenv import load_dotenv
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

def main():
    logging.info("Starting the training and evaluation process.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device selected for training: {device}")

    try:
        logging.info(f"Loading dataset from: {DATASET_DIR}")
        dataset = CustomDataset(root_dir=DATASET_DIR)
        num_classes = len(dataset.classes)
        logging.info(f"Number of classes detected: {num_classes}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # 1. Train SimCLR
    try:
        logging.info("Starting SimCLR training...")
        simclr_encoder = train_simclr(device)
        logging.info("SimCLR training complete. Proceeding to fine-tuning...")
        simclr_fine_tuned_model = fine_tune(simclr_encoder, device, num_classes)
        simclr_f1 = evaluate(simclr_fine_tuned_model, device, num_classes)
        logging.info(f"SimCLR F1 Score: {simclr_f1}")
    except Exception as e:
        logging.error(f"Error during SimCLR training or evaluation: {e}")

    # 2. Train BYOL
    try:
        logging.info("Starting BYOL training...")
        byol_encoder = train_byol(device)
        logging.info("BYOL training complete. Proceeding to fine-tuning...")
        byol_fine_tuned_model = fine_tune(byol_encoder, device, num_classes)
        byol_f1 = evaluate(byol_fine_tuned_model, device, num_classes)
        logging.info(f"BYOL F1 Score: {byol_f1}")
    except Exception as e:
        logging.error(f"Error during BYOL training or evaluation: {e}")

    # 3. Train DINO
    try:
        logging.info("Starting DINO training...")
        dino_encoder = train_dino(device)
        logging.info("DINO training complete. Proceeding to fine-tuning...")
        dino_fine_tuned_model = fine_tune(dino_encoder, device, num_classes)
        dino_f1 = evaluate(dino_fine_tuned_model, device, num_classes)
        logging.info(f"DINO F1 Score: {dino_f1}")
    except Exception as e:
        logging.error(f"Error during DINO training or evaluation: {e}")

    # 4. Train Baseline (no SSL)
    try:
        logging.info("Starting baseline training...")
        baseline_encoder = Encoder().to(device)
        baseline_fine_tuned_model = fine_tune(baseline_encoder, device, num_classes)
        baseline_f1 = evaluate(baseline_fine_tuned_model, device, num_classes)
        logging.info(f"Baseline F1 Score: {baseline_f1}")
    except Exception as e:
        logging.error(f"Error during baseline training or evaluation: {e}")
    
    # 5. Print Results
    try:
        logging.info("\nComparison of F1 Scores:")
        print(f"SimCLR: {simclr_f1}")
        print(f"BYOL: {byol_f1}")
        print(f"DINO: {dino_f1}")
        print(f"Baseline: {baseline_f1}")
    except NameError as e:
        logging.error(f"Error printing results, variable may not be defined: {e}")

if __name__ == "__main__":
    main()
