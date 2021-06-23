import torch
import torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    def __init__(self, classes, loaded_model):
        super(CNN, self).__init__()

        if loaded_model is False:
            self.cnn = models.resnet18(pretrained=False)
            self.cnn.fc = nn.Linear(512, classes)
        else:
            self.cnn = models.resnet18(pretrained=True)

            for param in self.cnn.parameters():
                param.requires_grad = False

            self.cnn.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.cnn(x)
