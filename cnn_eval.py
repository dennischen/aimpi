import os
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from alexnet import build_dataloader as build_alexnet_dataloader
from alexnet import build_net as build_alexnet
from googlenet import build_dataloader as build_googlenet_dataloader
from googlenet import build_net as build_googlenet
from lenet import build_dataloader as build_lenet_dataloader
from lenet import build_net as build_lenet
from nin import build_dataloader as build_nin_dataloader
from nin import build_net as build_nin
from resnet import build_dataloader as build_resnet_dataloader
from resnet import build_net as build_resnet
from utils import (get_fashion_mnist_labels, kmp_duplicate_lib_ok, savefig, show_images, try_gpu)
from vgg import build_dataloader as build_vgg_dataloader
from vgg import build_net as build_vgg


def main():
    kmp_duplicate_lib_ok()
    np.set_printoptions(linewidth=200, precision=3)
    torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

    num = 10

    def predict(case: str, net: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module],
                test_dataloader: data.DataLoader, wh: int, device: torch.device):
        X: torch.Tensor
        y: torch.Tensor
        X, y = next(iter(test_dataloader))

        X = X.to(device)
        y = y.to(device)
        net = net.to(device)

        labels = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
        hit, all = 0, 0
        for la, pr in zip(labels, preds):
            all += 1
            if (la == pr):
                hit = hit + 1
        print(f'{case} accuracy {hit/all:0.1f}')

        titles = [label + '\n' + pred for label, pred in zip(labels, preds)]
        n = len(X)
        show_images(X.reshape(n, wh, wh), 1, n, titles=titles)
        savefig(f'out/predict_{case}.png')

    cases: list[tuple[str, Callable[[], nn.Module], Callable[[int], tuple[data.DataLoader, data.DataLoader, int]]]] = []
    cases.append(('lenet', build_lenet, build_lenet_dataloader))
    cases.append(('alexnet', build_alexnet, build_alexnet_dataloader))
    cases.append(('googlenet', build_googlenet, build_googlenet_dataloader))
    cases.append(('resnet', build_resnet, build_resnet_dataloader))
    cases.append(('nin', build_nin, build_nin_dataloader))
    cases.append(('vgg', build_vgg, build_vgg_dataloader))

    device = try_gpu()

    print(f'use device {device}')

    for case, build_net, build_dataloader in cases:
        file = f'out/{case}.params'
        if (os.path.exists(file)):
            print(f'eval {case}')
            net = build_net()
            net.load_state_dict(torch.load(file))
            _, test_dataloader, wh = build_dataloader(num)
            predict(case, net, test_dataloader, wh, device)
        else:
            print(f'skip {case}, not training data')


if __name__ == '__main__':
    main()