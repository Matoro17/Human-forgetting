""" This script is intended to calculate mean and standard deviation (std) for 
the datasets used in the project. The references for this code were extracted 
from: https://www.kaggle.com/code/kozodoi/computing-dataset-mean-and-std,
https://kozodoi.me/blog/20210308/compute-image-stats and
https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/

 """
import os
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

import cv2
import sys
from tqdm import tqdm

sys.path.append('/home/alexsandro/pgcc/code/mestrado-alexsandro-defesa')
from datasets.symlinked_dataset import SymlinkedDataset

# device = torch.device('cpu')
num_workers = 1
# batch_size = 128
# image_size = 224
inception_image_size = 299

# baseline_path = '/home/alexsandro/data/dataset_cross-validation/baseline'
# rus_path = '/home/alexsandro/data/dataset_cross-validation/rus'
# ros_path = '/home/alexsandro/data/dataset_cross-validation/ros'

data_path = '/home/alexsandro/data/dataset_run/classes/RGB'



def get_mean_std(loader: DataLoader):
    n_pixels = 0
    mean = 0.0
    std = 0.0
    mean_sum = 0.0
    std_sum = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        n_pixels += batch_size * height * width
        # print(f"Normal Shape: {images.shape}")
        batch_sum = torch.sum(images, dim=[0, 2, 3])  # Sum of pixel values for each channel
        batch_sum_squared = torch.sum(images ** 2, dim=[0, 2, 3]) 

        mean_sum += batch_sum
        std_sum += batch_sum_squared
    
    mean = mean_sum / n_pixels
    std = torch.sqrt(std_sum / n_pixels - mean ** 2)

    # print(f"Mean sum: {mean_sum}, Pixel sum: {n_pixels}")
    # print(f"Std sum: {std_sum}")
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")       
    
    
    return mean, std


# Mean sum: tensor([1.5367e+08, 1.1911e+08, 1.3683e+08]), Pixel sum: 220222464
# Std sum: tensor([1.1325e+08, 7.2890e+07, 9.0320e+07])
# Mean: tensor([0.6978, 0.5408, 0.6213])
# Std: tensor([0.1654, 0.1961, 0.1552])
# Mean and Std for (224, 224): tensor([0.6978, 0.5408, 0.6213]), tensor([0.1654, 0.1961, 0.1552])
# Mean sum: tensor([2.7380e+08, 2.1222e+08, 2.4380e+08]), Pixel sum: 392380989
# Std sum: tensor([2.0204e+08, 1.3023e+08, 1.6115e+08])
# Mean: tensor([0.6978, 0.5408, 0.6213])
# Std: tensor([0.1673, 0.1985, 0.1570])
# Mean and Std for (299, 299): tensor([0.6978, 0.5408, 0.6213]), tensor([0.1673, 0.1985, 0.1570])

def mean_std(data_path, input_size):
    # print("Calling mean_std")
    data_transform = transforms.Compose([
                    transforms.Resize(size=input_size),                
                    transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(root=data_path,
                                            transform=data_transform,
                                            target_transform=None)
    
    data_loader = DataLoader(dataset=dataset,
                                        batch_size=32,
                                        num_workers=num_workers)
    
    return get_mean_std(data_loader)


def mean_std_for_symlinks(data_path, input_size):
    # print("Calling mean_std")
    data_transform = transforms.Compose([
                    transforms.Resize(size=input_size),                
                    transforms.ToTensor()])
    
    dataset = SymlinkedDataset(data_dir=data_path,
                                            transform=data_transform)
    
    data_loader = DataLoader(dataset=dataset,
                                        batch_size=32,
                                        num_workers=num_workers)
    
    return get_mean_std(data_loader)

def main():
    input_size = (224, 224)

    for input_size in [(224, 224), (299, 299)]:
        # data_transform = transforms.Compose([
        #             transforms.Resize(size=input_size),                
        #             transforms.ToTensor()])
        
        # dataset = datasets.ImageFolder(root=data_path,
        #                                     transform=data_transform,
        #                                     target_transform=None)
        
        # data_loader = DataLoader(dataset=dataset,
        #                                 batch_size=128,
        #                                 num_workers=os.cpu_count())
        
        # mean, std = get_mean_std(data_loader)
        mean, std = mean_std(data_path=data_path, input_size=input_size)

        print(f"Mean and Std for {input_size}: {mean}, {std}")
    
# if __name__ == '__main__':
#     main()

