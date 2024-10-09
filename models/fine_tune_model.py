import torch.nn as nn

class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder  # Use the pretrained encoder
        self.fc = nn.Linear(512, num_classes)  # Add a final fully connected layer for classification

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
