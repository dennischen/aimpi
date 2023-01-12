import math
from typing import Callable, Union

import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

from utils import (basename_noext, kmp_duplicate_lib_ok, savefig, synthetic_data, evaluate_loss, linreg, squared_loss,
                   sgd, load_data, load_data_fashion_mnist, train_ani)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)


def dropout_layer(X: torch.Tensor, dropout: float):
    assert 0 <= dropout <= 1
    # 在本情況中,所有元素都被丟棄
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情況中,所有元素都被保留
    if dropout == 0:
        return X
    # 0. or 1.
    mask = (torch.rand(X.shape) > dropout).float()
    # print(f'>>>mask {dropout} {mask}')
    return mask * X / (1.0 - dropout)


X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(f'X {X}, {X.sum()}')
DL = dropout_layer(X, 0)
print(f'X 0.0 {DL}, {DL.sum()}')
DL = dropout_layer(X, 0.25)
print(f'X 0.25 {DL}, {DL.sum()}')
DL = dropout_layer(X, 0.5)
print(f'X 0.5 {DL}, {DL.sum()}')
DL = dropout_layer(X, 0.75)
print(f'X 0.75 {DL}, {DL.sum()}')
DL = dropout_layer(X, 1)
print(f'X 1.0 {DL}, {DL.sum()}')

num_inputs, num_hiddens1, num_hiddens2, num_outputs = 784, 256, 256, 10

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, num_hiddens1: int, num_hiddens2: int, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在訓練模型時才使用dropout
        if self.training:
            # 在第一個全連接層之後新增一個dropout層
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二個全連接層之後新增一個dropout層
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ani(net, train_dataloader, test_dataloader, loss, num_epochs, trainer)

savefig(f'out/{basename_noext(__file__)}.png')

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    # 在第一個全連接層之後新增一個dropout層
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    # 在第二個全連接層之後新增一個dropout層
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_ani(net, train_dataloader, test_dataloader, loss, num_epochs, trainer)
savefig(f'out/{basename_noext(__file__)}_concise.png')