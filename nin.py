import numpy as np
import torch
from torch import nn

from utils import (basename_noext, kmp_duplicate_lib_ok, load_data_fashion_mnist, savefig, train_ch6, try_gpu)


def build_net():
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 標籤類別數是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 將四維的輸出轉成二維的輸出,其形狀為(批次大小,10)
        nn.Flatten())
    return net


def build_dataloader(batch_size: int):
    wh = 224
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

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_dataloader, test_dataloader, _ = build_dataloader(batch_size)
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())

    savefig(f'out/{basename_noext(__file__)}.png')
    torch.save(net.state_dict(), f'out/{basename_noext(__file__)}.params')


if __name__ == '__main__':
    main()