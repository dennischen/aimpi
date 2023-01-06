import math
from typing import Callable, Union

import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

from utils import (basename_noext, kmp_duplicate_lib_ok, savefig, train_epoch, load_data, Animator)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=120)

max_degree = 20  # 多項式的最大階數
n_train, n_test = 100, 100  # 訓練和測試資料集大小
true_w = np.zeros(max_degree)  # 分配大量的空間
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
print(f'true_w {true_w}')

# (200, 1)
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
print(f'features {features.shape}, {features}')

# (1, 20)
poly = np.arange(max_degree).reshape(1, -1)
print(f'poly {poly.shape}, {poly}')

# (200, 20)
poly_features = np.power(features, poly)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# poly_feature = x ** i / !i
print(f'poly_features {poly_features.shape}, {poly_features}')

# (200,20)@(20,) = (200,)
labels = np.dot(poly_features, true_w)
# add bias
labels += np.random.normal(scale=0.1, size=labels.shape)
print(f'labels b {labels.shape}, {labels}')

# ndarray to tensor
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]
]

print(f'true_w {true_w}')
print(f'features {features}')
print(f'poly_features {poly_features}')
print(f'labels {labels}')


def evaluate_loss(net: torch.nn.Module, dataloader: data.DataLoader, loss: Callable[[torch.Tensor, torch.Tensor],
                                                                                    torch.Tensor]):
    """評估給定資料集上模型的損失"""
    metric = d2l.Accumulator(2)  # 損失的總和,樣本數量
    X: torch.Tensor
    y: torch.Tensor
    for X, y in dataloader:
        out: torch.Tensor = net(X)
        y = y.reshape(out.shape)
        lo = loss(out, y)
        metric.add(lo.sum(), lo.numel())
    return metric[0] / metric[1]


def train(train_features: torch.Tensor,
          test_features: torch.Tensor,
          train_labels: torch.Tensor,
          test_labels: torch.Tensor,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不設定偏置，因為我們已經在多項式中實現了它
    l1 = nn.Linear(input_shape, 1, bias=False)
    net = nn.Sequential(l1)
    batch_size = min(10, train_labels.shape[0])
    train_dataloader = load_data((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_dataloader = load_data((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch',
                        ylabel='loss',
                        yscale='log',
                        xlim=[1, num_epochs],
                        ylim=[1e-3, 1e2],
                        legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch(net, train_dataloader, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_dataloader, loss), evaluate_loss(net, test_dataloader, loss)))
    print('weight:', l1.weight.data.numpy())

print('true weight:', true_w.data.numpy())
# poly_features = [n_train + n_test, max_degree]
# 從多項式特徵中選擇前4個維度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
savefig(f'out/{basename_noext(__file__)}.png')

# 從多項式特徵中選擇前2個維度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
savefig(f'out/{basename_noext(__file__)}_underfit.png')

# 從多項式特徵中選取所有維度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
savefig(f'out/{basename_noext(__file__)}_overfit.png')