import torch.nn as nn
import torch

class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        
        # Freeze encoder parameters to avoid retraining them
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Determine the output features from the encoder and adjust the input to `fc`
        self.encoder_output_size = self._get_encoder_output_size()
        self.fc = nn.Linear(self.encoder_output_size, num_classes)

    def _get_encoder_output_size(self):
        # Move dummy input to the same device as the encoder
        dummy_input = torch.randn(1, 3, 224, 224).to(next(self.encoder.parameters()).device)
        with torch.no_grad():
            output = self.encoder(dummy_input)
        return output.shape[1]

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
