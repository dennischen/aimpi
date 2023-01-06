import math
from typing import Callable, Union

import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

from utils import (basename_noext, kmp_duplicate_lib_ok, savefig, synthetic_data, evaluate_loss, linreg, squared_loss,
                   sgd, load_data, Animator)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=120)

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_dataloader = load_data(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_dataloader = load_data(test_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w: torch.Tensor):
    return torch.sum(w.pow(2)) / 2


def train(lambd: int):
    w, b = init_params()

    def net(X):
        return linreg(X, w, b)

    loss = squared_loss

    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        X: torch.tensor
        y: torch.tensor
        for X, y in train_dataloader:
            # 增加了L2范數懲罰項，
            # 廣播機制使l2_penalty(w)成為一個長度為batch_size的向量
            lo: torch.Tensor = loss(net(X), y) + lambd * l2_penalty(w)
            lo.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_dataloader, loss), evaluate_loss(net, test_dataloader, loss)))
    print('w的L2范數是：', torch.norm(w).item())


def train_concise(weight_decay: int):
    l1 = nn.Linear(num_inputs, 1)
    net = nn.Sequential(l1)
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置參數沒有衰減
    trainer = torch.optim.SGD([{"params": l1.weight, 'weight_decay': weight_decay}, {"params": l1.bias}], lr=lr)
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        X: torch.Tensor
        y: torch.Tensor
        for X, y in train_dataloader:
            trainer.zero_grad()
            lo: torch.Tensor = loss(net(X), y)
            lo.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_dataloader, loss), evaluate_loss(net, test_dataloader, loss)))
    print('w的L2范數：', l1.weight.norm().item())


train(0)
savefig(f'out/{basename_noext(__file__)}_0.png')
train(3)
savefig(f'out/{basename_noext(__file__)}_3.png')

train_concise(0)
savefig(f'out/{basename_noext(__file__)}_concise_0.png')
train_concise(3)
savefig(f'out/{basename_noext(__file__)}_concise_3.png')