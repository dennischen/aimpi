import math
from typing import Callable, Union

import numpy as np
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

from utils import (basename_noext, kmp_duplicate_lib_ok, savefig, synthetic_data, evaluate_loss, linreg, squared_loss,
                   sgd, load_data, load_data_fashion_mnist, train_ani, plot)

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

savefig(f'out/{basename_noext(__file__)}_lost.png')

M = torch.normal(0, 1, size=(4, 4))
print('一個矩陣 \n', M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('乘以100個矩陣後\n', M)