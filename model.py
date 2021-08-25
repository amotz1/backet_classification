import torch
import torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.cnn(x)


class Lenet5(nn.Module):
    def __init__(self, classes):
        super(Lenet5, self).__init__()
#         declare layers
        self.cnn1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.cnn3(x)
        x = self.relu3(x)
        x = x.reshape(x.shape[0], -1)

        x = self.linear1(x)
        x = self.linear2(x)

        return x

#         declare forward pass (how model influence the tensors)

