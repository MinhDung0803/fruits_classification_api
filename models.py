from torchvision import models
import torch.nn as nn


class ResNet_model(nn.Module):
  def __init__(self, number_of_classes):
    super().__init__()
    self.network = models.resnet18(pretrained=True)
    features_in = self.network.fc.in_features
    self.network.fc = nn.Linear(features_in, number_of_classes)

  def forward(self, img):
    return self.network(img)



