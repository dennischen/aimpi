import math
import multiprocessing
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
from d2l import torch as d2l
from matplotlib.axes import Axes, Subplot
from matplotlib.figure import Figure, FigureBase, figaspect
from torch.utils import data
from torchvision import transforms

from utils import (basename_noext, get_fashion_mnist_labels, kmp_duplicate_lib_ok, load_data_fashion_minist)

kmp_duplicate_lib_ok()

batch_size = 256
lr = 0.1

train_dataloader, test_dataloader = load_data_fashion_minist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


######################################
def softmax(X: torch.Tensor):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def test_softmax():
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(f'softmax {X},\nprob {X_prob},\nsum is {X_prob.sum(1)}')


test_softmax()


######################################
def net(X: torch.Tensor):
    """The net"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


######################################
def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor):
    """"The lose"""
    return -torch.log(y_hat[range(len(y_hat)), y])


def test_cross_entropy():
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(f'cross_entropy {cross_entropy(y_hat, y)}')


test_cross_entropy()

######################################


def count_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """計算預測正確的數量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def test_count_accuracy():
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(f'accuracy count {count_accuracy(y_hat, y)}, accuracy {count_accuracy(y_hat, y)/len(y)}')


test_count_accuracy()


class Accumulator:  #@save
    """在n個變數上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module], dataloader: data.DataLoader):
    """計算在指定資料集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 將模型設定為評估模式
    metric = Accumulator(2)  # 正確預測數、預測總數
    with torch.no_grad():
        X: torch.Tensor
        y: torch.Tensor
        for X, y in dataloader:
            metric.add(count_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def test_evaluate_accuracy():
    init_acc = evaluate_accuracy(net, test_dataloader)
    print(f'evaluate_accuracy on init net by test data {init_acc}')


test_evaluate_accuracy()


def train_epoch(net: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module], dataloader: data.DataLoader,
                loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], updater: Union[Callable[[float, int], None],
                                                                                           torch.optim.Optimizer]):
    """訓練模型一個迭代週期"""
    # 將模型設定為訓練模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 訓練損失總和、訓練精準度總和、樣本數
    metric = Accumulator(3)
    X: torch.Tensor
    y: torch.Tensor
    for X, y in dataloader:
        # 計算梯度並更新參數
        y_hat = net(X)
        lo = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch內建的最佳化器和損失函數
            updater.zero_grad()
            lo.mean().backward()
            updater.step()
        else:
            # 使用定製的最佳化器和損失函數
            lo.sum().backward()
            updater(X.shape[0])
        metric.add(float(lo.sum()), count_accuracy(y_hat, y), y.numel())
    # 返回訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]


def updater(batch_size: int):
    return d2l.sgd([W, b], lr, batch_size)


class Animator:
    """在動畫中繪製資料"""
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地繪製多條線
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函數捕獲參數
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()

    def add(self, x, y):
        # 向圖表中新增多個資料點
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        self.axes[0].relim()  # recompute the data limits
        self.axes[0].autoscale_view()  # automatic axis scaling
        self.fig.canvas.flush_events()


def train(net: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module], train_dataloader: data.DataLoader,
          test_dataloader: data.DataLoader, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], num_epochs: int,
          updater: Union[Callable[[float, int], None], torch.optim.Optimizer]):
    """訓練模型"""
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_dataloader, loss, updater)
        test_acc = evaluate_accuracy(net, test_dataloader)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
        train_loss, train_acc = train_metrics
        print(f'Train loss in {epoch} = {train_loss:.3f}')
        print(f'Train accuarcy in {epoch} = {train_acc:.3f}')
        print(f'Test accuarcy in {epoch} = {test_acc:.3f}')

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epochs = 10
train(net, train_dataloader, test_dataloader, cross_entropy, num_epochs, updater)

plt.savefig(f'out/{basename_noext(__file__)}_training.png')


def show_images(imgs: torch.Tensor, num_rows: int, num_cols: int, titles: tuple[str] = None, scale=1.5):
    """繪製圖像列表"""
    figsize = (num_cols * scale, (num_rows + 1) * scale)
    axes_arr: numpy.ndarray[Axes]
    _, axes_arr = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes_arr = axes_arr.flatten()
    axes: Axes
    for i, (axes, img) in enumerate(zip(axes_arr, imgs)):
        if torch.is_tensor(img):
            pixels = img.numpy()  # (28, 28)
            axes.imshow(pixels)
        else:
            # PIL圖片
            axes.imshow(img)
        axes.axes.get_xaxis().set_visible(False)
        axes.axes.get_yaxis().set_visible(False)
        if titles:
            axes.set_title(titles[i])
            print(f'>>{titles[i]}<<')
    return axes_arr


def predict(net: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module], test_dataloader: data.DataLoader, n=8):
    """預測標籤"""
    n = min(n, batch_size)
    X: torch.Tensor
    y: torch.Tensor
    X, y = next(iter(test_dataloader))
    labels = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [label + '\n' + pred for label, pred in zip(labels, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.savefig(f'out/{basename_noext(__file__)}_pred.png')


predict(net, test_dataloader)