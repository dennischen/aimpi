from typing import Callable, Union

import torch
import numpy as np
from torch import nn
from torch.utils import data

from utils import (Accumulator, Animator, Timer, basename_noext, count_accuracy, kmp_duplicate_lib_ok,
                   load_data_fashion_mnist, savefig, try_gpu, train_ch6)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1,1,28,28 -> 1,6,28,28 (28+2+2-5+1=28)
    nn.Sigmoid(),  # 1,6,28,28
    nn.AvgPool2d(kernel_size=2, stride=2),  # 1,6,28,28 -> 1,6,14,14 (28/2=14)
    nn.Conv2d(6, 16, kernel_size=5),  # 1,6,14,14 -> 1,16,10,10 (14-5+1=10)
    nn.Sigmoid(),  # 1,16,10,10
    nn.AvgPool2d(kernel_size=2, stride=2),  # 1,16,5,5 (10/2=2)
    nn.Flatten(),  # 1,400
    nn.Linear(16 * 5 * 5, 120),  # 1,120
    nn.Sigmoid(),  # 1,120
    nn.Linear(120, 84),  # 1,84
    nn.Sigmoid(),  # 1,84
    nn.Linear(84, 10))  # 1,10

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

def evaluate_accuracy_gpu(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                          data_loader: data.DataLoader,
                          device: torch.device = None):
    """使用GPU計算模型在資料集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 設定為評估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正確預測的數量，總預測的數量
    metric = Accumulator(2)
    with torch.no_grad():
        X: torch.Tensor
        y: torch.Tensor
        for X, y in data_loader:
            if isinstance(X, list):
                # BERT微調所需的（之後將介紹）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(count_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module], train_dataloader: data.DataLoader,
          test_dataloader: data.DataLoader, num_epochs: int, lr: float, device: torch.device):
    """用GPU訓練模型(在第六章定義)"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'],
                        figsize=(5, 5))
    timer, num_batches = Timer(), len(train_dataloader)
    for epoch in range(num_epochs):
        # 訓練損失之和，訓練精準率之和，樣本數
        metric = Accumulator(3)
        net.train()  # trun to train mode (because evaluate_accuracy_gpu will trun it to eval)
        X: torch.Tensor
        y: torch.Tensor
        for i, (X, y) in enumerate(train_dataloader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat: torch.Tensor = net(X)
            lo: torch.Tensor = loss(y_hat, y)
            lo.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(lo * X.shape[0], count_accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_dataloader)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


batch_size = 256
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.9, 10
train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
savefig(f'out/{basename_noext(__file__)}.png')