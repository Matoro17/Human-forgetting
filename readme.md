# Self-Supervised Learning with PyTorch

This project demonstrates self-supervised learning with PyTorch using CIFAR-10 dataset.

## Structure

- `data/`: Contains data loading scripts.
- `models/`: Contains model definitions.
- `utils/`: Contains utility functions such as data augmentations.
- `train_pretext.py`: Script to train the model on a pretext task (rotation prediction).
- `train_classifier.py`: Script to fine-tune the model on the classification task.

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt