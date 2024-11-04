import torch

def save_model(model, optimizer, epoch, path):
    """
    Save the model's state dictionary, optimizer state, and the current epoch.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): The current epoch number.
        path (str): The file path to save the model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved at {path}")
