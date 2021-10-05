import torch
import torch.nn as nn
from torchvision import models


def crop_fm(fm, tfm):
    target_size = tfm.size()[2]
    tensor_size = fm.size()[2]

    delta_size = tensor_size - target_size
    delta_size = delta_size//2

    return fm[:, :, delta_size: tensor_size-delta_size, delta_size: tensor_size-delta_size]


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
        self.cnn1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.batch1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.batch3 = nn.BatchNorm2d(120)
        self.relu3 = nn.ReLU()
        self.linear1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(84, classes)

    def forward(self, x):
        # x = self.cnn1(x)
        # x = self.batch1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)
        x = self.maxpool1(self.relu1(self.batch1(self.cnn1(x))))
        #
        # x = self.cnn2(x)
        # x = self.batch2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        x = self.maxpool2(self.relu2(self.batch2(self.cnn2(x))))

        # x = self.cnn3(x)
        # x = self.batch3(x)
        # x = self.relu3(x)
        x = self.relu3(self.batch3(self.cnn3(x)))

        x = x.reshape(x.shape[0], -1)

        # x = self.linear1(x)
        # x = self.relu4(x)
        # x = self.linear2(x)
        x = self.linear2(self.relu4(self.linear1(x)))

        return x

#         declare forward pass (how model influence the tensors)

class FullyConnected(nn.Module):
    def __init__(self, classes):
        super(FullyConnected, self).__init__()
        self.linear1 = nn.Linear(3*6*64, 300)
        self.batch1 = nn.BatchNorm1d(300)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(300, 100)
        self.batch2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu1(self.batch1(self.linear1(x)))
        x = self.relu2(self.batch2(self.linear2(x)))

        return x


class UnetDownSampleLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(UnetDownSampleLayer, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_chan, out_chan,  kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv2d(out_chan, out_chan,  kernel_size=3),
                                        nn.ReLU())

    def forward(self, x):
        x = self.conv_block(x)
        return x


class UnetUpSampleLayer(nn.Module):
    def __init__(self,in_chans, out_chans):
        super(UnetUpSampleLayer, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.transpose_conv(x)
        return x


class UnetDoubleConvLayer:
    def __init__(self, in_chan, out_chan):
        super(UnetDoubleConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, int(out_chan/2), kernel_size=3)
        self.conv2 = nn.Conv2d(int(out_chan/2), int(out_chan/2), kernel_size=3)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.maxpool = nn.MaxPool(kernel_size=2, stride=2)
        self.UnetDownSampleLayer1 = UnetDownSampleLayer(1, 64)
        self.UnetDownSampleLayer2 = UnetDownSampleLayer(64, 128)
        self.UnetDownSampleLayer3 = UnetDownSampleLayer(128, 256)
        self.UnetDownSampleLayer4 = UnetDownSampleLayer(256, 512)
        self.UnetDownSampleLayer5 = UnetDownSampleLayer(512, 1024)

        self.UnetUpSampleLayer1 = UnetUpSampleLayer(1024, 512)
        self.conv1 = UnetDoubleConvLayer(1024, 512)
        self.UnetUpSampleLayer2 = UnetUpSampleLayer(512, 256)
        self.conv2 = UnetDoubleConvLayer(512, 256)
        self.UnetUpSampleLayer3 = UnetUpSampleLayer(256, 128)
        self.conv3 = UnetDoubleConvLayer(256, 128)
        self.UnetUpSampleLayer4 = UnetUpSampleLayer(128, 64)
        self.conv4 = UnetDoubleConvLayer(128, 64)
        self.conv5 = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):

        x1 = self.UnetDownSampleLayer1(image)
        x2 = self.maxpool(x1)
        x2 = self.UnetDownSampleLayer2(x2)
        x3 = self.maxpool(x2)
        x3 = self.UnetDownSampleLayer3(x3)
        x4 = self.maxpool(x3)
        x4 = self.UnetDownSampleLayer4(x4)
        x5 = self.maxpool(x4)
        x5 = self.UnetDownSampleLayer5(x5)
        print(x5.shape)
        x6 = self.UnetUpSampleLayer1(x5)
        y = crop_fm(x4, x6)
        x7 = torch.cat([y, x6],1)
        x7 = self.conv1(x7)
        x8 = self.UnetUpSampleLayer2(x7)
        y1 = crop_fm(x3, x8)
        print("x8.shape ",x8.shape)
        x9 = torch.cat([y1,x8],1)
        x9 = self.conv2(x9)
        x10 = self.UnetUpSampleLayer3(x9)
        print("x9.shape", x9.shape)
        y2 = crop_fm(x2, x10)
        x11 = torch.cat([y2, x10],1)
        print("x11.shape ",x11.shape)
        x11 = self.conv3(x11)
        x12 = self.UnetUpSampleLayer4(x11)
        y3 = crop_fm(x1, x12)
        x13 = torch.cat([y3, x12],1)
        output = self.conv4(x13)
        output = self.conv5(output)
        print("output.sshape ", output.shape)

        return output







