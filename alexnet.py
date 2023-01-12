import torch
from torch import nn
import numpy as np
from utils import (basename_noext, kmp_duplicate_lib_ok, load_data_fashion_mnist, savefig, try_gpu, train_ch6)


def build_net():
    return nn.Sequential(
        # 這裡使用一個11*11的更大窗口來捕捉對象。
        # 同時,步幅為4,以減少輸出的高度和寬度。
        # 另外,輸出通道的數目遠大於LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 減小摺積窗口,使用填充為2來使得輸入與輸出的高和寬一致,且增大輸出通道數
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三個連續的摺積層和較小的摺積窗口。
        # 除了最後的摺積層,輸出通道的數量進一步增加。
        # 在前兩個摺積層之後,匯聚層不用於減少輸入的高度和寬度
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 這裡,全連接層的輸出數量是LeNet中的好幾倍。使用dropout層來減輕過擬合
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最後是輸出層。由於這裡使用Fashion-MNIST,所以用類別數為10,而非論文中的1000
        nn.Linear(4096, 10))


def build_dataloader(batch_size: int):
    wh = 224
    return (*load_data_fashion_mnist(batch_size, resize=wh), wh)


def main():

    kmp_duplicate_lib_ok()
    np.set_printoptions(linewidth=200, precision=3)
    torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

    net = build_net()

    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 256
    train_dataloader, test_dataloader,_ = build_dataloader(batch_size)
    lr, num_epochs = 0.01, 10
    train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
    savefig(f'out/{basename_noext(__file__)}.png')
    torch.save(net.state_dict(), f'out/{basename_noext(__file__)}.params')


if __name__ == '__main__':
    main()