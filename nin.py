import torch
import numpy as np
from torch import nn
from utils import (Accumulator, Animator, Timer, basename_noext, count_accuracy, kmp_duplicate_lib_ok,
                   load_data_fashion_mnist, savefig, try_gpu, train_ch6)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

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
    # 將四維的輸出轉成二維的輸出，其形狀為(批次大小,10)
    nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

savefig(f'out/{basename_noext(__file__)}.png')