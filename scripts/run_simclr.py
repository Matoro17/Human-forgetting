import torch
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from training.train_simclr import train_simclr

import os
from dotenv import load_dotenv  # Import dotenv to load .env files

# Load environment variables from .env file
load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "datasets/train")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder()
    train_set = CustomDataset(root_dir=DATASET_DIR, split='train')
    train_simclr(encoder, train_set, device)
