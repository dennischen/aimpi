import torch
import numpy as np
from torch import nn
from utils import (Accumulator, Animator, Timer, basename_noext, count_accuracy, kmp_duplicate_lib_ok,
                   load_data_fashion_mnist, savefig, try_gpu, train_ch6)
from torch.nn import functional as F


def build_net():
    class Residual(nn.Module):
        def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)


    # X = torch.rand(4, 3, 6, 6)
    # blk = Residual(3, 3)
    # Y = blk(X)
    # print(Y.shape)

    # blk = Residual(3, 6, use_1x1conv=True, strides=2)
    # Y = blk(X)
    # print(Y.shape)

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),  #
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk


    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 10)  #
    )
    return net


def build_dataloader(batch_size: int):
    wh = 96
    return (*load_data_fashion_mnist(batch_size, resize=wh), wh)



def main():

    kmp_duplicate_lib_ok()
    np.set_printoptions(linewidth=200, precision=3)
    torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

    net = build_net()

    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_dataloader, test_dataloader,_ = build_dataloader(batch_size)
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())

    savefig(f'out/{basename_noext(__file__)}_concise.png')
    torch.save(net.state_dict(), f'out/{basename_noext(__file__)}.params')


if __name__ == '__main__':
    main()