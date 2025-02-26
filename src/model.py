import torch.nn as nn
from torchvision import models

class AgeModel(nn.Module):
    def __init__(self, model):
        super(AgeModel, self).__init__()
        if model == 'resnet34':
            self.model = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        elif model == 'resnet101':
            self.model = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")
        elif model == 'resnet18':
            self.model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        elif model == 'resnet50':
            self.model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        elif model == 'swin_v2_s':
            self.model = models.swin_v2_s(weights='IMAGENET1K_V1')
        elif model == 'swin_v2_t':
            self.model = models.swin_v2_t(weights='IMAGENET1K_V1')
        elif model == 'swin_v2_b':
            self.model = models.swin_v2_b(weights='IMAGENET1K_V1')
        elif model == 'convnext_small':
            self.model = models.convnext_small(weights='IMAGENET1K_V1')
        elif model == 'convnext_tiny':
            self.model = models.convnext_tiny(weights='IMAGENET1K_V1')
        elif model == 'vit_b_16':
            self.model = models.vit_b_16(weights='IMAGENET1K_V1')

        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x