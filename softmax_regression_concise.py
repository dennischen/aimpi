from typing import Callable, Union

import torch
from d2l import torch as d2l
from matplotlib.axes import Axes
from torch import nn
from torch.utils import data

from utils import (basename_noext, get_fashion_mnist_labels,
                   kmp_duplicate_lib_ok, load_data_fashion_minist, train, predict, savefig)



kmp_duplicate_lib_ok()

batch_size = 256
lr = 0.1

train_dataloader, test_dataloader = load_data_fashion_minist(batch_size)

# Flatten makes 28x28 to 784
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_linear)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=lr)

num_epochs = 10

train(net, train_dataloader, test_dataloader, loss, num_epochs, trainer)

savefig(f'out/{basename_noext(__file__)}_train.png')

predict(net, test_dataloader)

savefig(f'out/{basename_noext(__file__)}_pred.png')