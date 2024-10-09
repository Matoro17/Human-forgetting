import torchvision.transforms as transforms

class SimCLRTransform:
    def __init__(self, s=0.5):
        self.random_crop = transforms.RandomResizedCrop(size=224)
        self.color_distortion = transforms.Compose([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3)
        ])
        self.to_tensor = transforms.ToTensor()  # Convert images to tensors

    def __call__(self, x):
        x_i = self.to_tensor(self.color_distortion(self.random_crop(x)))
        x_j = self.to_tensor(self.color_distortion(self.random_crop(x)))
        return x_i, x_j

# Wrapper dataset for SimCLR
class SimCLRDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]  # Retrieve the image and ignore the label
        x_i, x_j = self.transform(x)  # Apply the SimCLR transformation
        return x_i, x_j

# Wrapper dataset for BYOL
class BYOLDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = SimCLRTransform()  # Use similar transformations to SimCLR

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]  # Retrieve the image and ignore the label
        x_i, x_j = self.transform(x)  # Apply the BYOL transformation
        return x_i, x_j