import torch
from torch import nn, optim
from data.load_data import get_data_loader
from models.classifier_model import CIFAR10Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = get_data_loader()

num_classes = 10
model = CIFAR10Classifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Fine-Tuning')