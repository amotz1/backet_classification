import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, classes, model ='resnet18'):
        super(CNN, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.cnn(x)
