import torch
import numpy as np
from torch import nn
from utils import (Accumulator, Animator, Timer, basename_noext, count_accuracy, kmp_duplicate_lib_ok,
                   load_data_fashion_mnist, savefig, try_gpu, train_ch6)
from torch.nn import functional as F

kmp_duplicate_lib_ok()
np.set_printoptions(linewidth=200, precision=3)
torch.set_printoptions(linewidth=200, sci_mode=False, precision=3)


def batch_norm(X: torch.Tensor, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通過is_grad_enabled來判斷當前模式是訓練模式還是預測模式
    if not torch.is_grad_enabled():
        # 如果是在預測模式下,直接使用傳入的移動平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全連接層的情況,計算特徵維上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            # 使用二維摺積層的情況,計算通道維上（axis=1）的均值和方差。
            # 這裡我們需要保持X的形狀以便後面可以做廣播運算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # 訓練模式下,用當前的均值和方差做標準化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移動平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 縮放和移位
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features：完全連接層的輸出數量或摺積層的輸出通道數。
    # num_dims：2表示完全連接層,4表示摺積層
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 參與求梯度和迭代的拉伸和偏移參數,分別初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型參數的變數初始化為0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在記憶體上,將moving_mean和moving_var
        # 複製到X所在視訊記憶體上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新過的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X,
                                                          self.gamma,
                                                          self.beta,
                                                          self.moving_mean,
                                                          self.moving_var,
                                                          eps=1e-5,
                                                          momentum=0.9)
        return Y


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    nn.Sigmoid(),  #
    nn.AvgPool2d(kernel_size=2, stride=2),  #
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    nn.Sigmoid(),  #
    nn.AvgPool2d(kernel_size=2, stride=2),  #
    nn.Flatten(),  #
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, num_dims=2),
    nn.Sigmoid(),  #
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2),
    nn.Sigmoid(),  #
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
savefig(f'out/{basename_noext(__file__)}.png')

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    nn.BatchNorm2d(6),
    nn.Sigmoid(),  #
    nn.AvgPool2d(kernel_size=2, stride=2),  #
    nn.Conv2d(6, 16, kernel_size=5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),  #
    nn.AvgPool2d(kernel_size=2, stride=2),  #
    nn.Flatten(),  #
    nn.Linear(256, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),  #
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),  #
    nn.Linear(84, 10))

train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, try_gpu())
savefig(f'out/{basename_noext(__file__)}_concise.png')