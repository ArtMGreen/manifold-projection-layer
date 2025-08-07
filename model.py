import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim)

    def forward(self, x):
        z_prime = self.fc(x)
        energy = torch.mean(z_prime ** 2, dim=1, keepdim=True)
        return z_prime, energy

class ModelWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.model_backbone = backbone
        self.proj_head = ProjHead(512, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.model_backbone(x)
        z_prime, energy = self.proj_head(features)
        logits = self.classifier(z_prime)
        return logits, energy
