import torch
from models.encoder import Encoder
from datasets.custom_dataset import CustomDataset
from training.train_simclr import train_simclr

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder()
    train_set = CustomDataset(root_dir='../datasetMestradoGledson+gabriel', split='train')
    train_simclr(encoder, train_set, device)
