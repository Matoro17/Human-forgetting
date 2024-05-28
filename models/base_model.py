import torchvision.models as models

def load_base_model():
    base_model = models.resnet18(pretrained=True)
    return base_model