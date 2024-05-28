import torch
from torch import nn, optim
from data.load_data import get_data_loader
from models.rotation_model import RotationPredictionModel
from utils.augmentations import generate_rotation_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = get_data_loader()

model = RotationPredictionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        rotated_images, rotation_labels = generate_rotation_batch(batch)
        rotated_images, rotation_labels = rotated_images.to(device), rotation_labels.to(device)

        optimizer.zero_grad()
        outputs = model(rotated_images)
        loss = criterion(outputs, rotation_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')