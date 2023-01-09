import math
import os
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import torch.nn as nn
from matplotlib.axes import Axes
from torch.utils import data
from torchvision import transforms


def kmp_duplicate_lib_ok(on: bool = True):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" if on else "FALSE"


def basename_noext(p: str):
    return os.path.splitext(os.path.basename(p))[0]


def synthetic_data(w: torch.Tensor, b: torch.Tensor, num_examples: int):
    """Generate y = Xw + b + noise.

    Defined in :numref:`sec_utils`"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, torch.reshape(y, (-1, 1))


def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """The linear regression model.

    Defined in :numref:`sec_utils`"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """Squared loss.

    Defined in :numref:`sec_utils`"""
    return (y_hat - torch.reshape(y, y_hat.shape))**2 / 2


def evaluate_loss(net: nn.Module, dataloader: data.DataLoader, loss: Callable[[torch.Tensor, torch.Tensor],
                                                                              torch.Tensor]):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    X: torch.Tensor
    y: torch.Tensor
    for X, y in dataloader:
        out = net(X)
        y = torch.reshape(y, out.shape)
        lo = loss(out, y)
        metric.add(lo.sum(), lo.numel())
    return metric[0] / metric[1]


def sgd(params: tuple[torch.Tensor], lr: float, batch_size: int):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def load_data_fashion_minist(batch_size: int):
    trans = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root="download-data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="download-data", train=False, transform=trans, download=True)

    train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST資料集的文字標籤"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data(data_arrays: torch.Tensor, batch_size: int, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


class Accumulator:
    """在n個變數上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def count_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """計算預測正確的數量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                      dataloader: data.DataLoader,
                      device="cpu"):
    """計算在指定資料集上模型的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 將模型設定為評估模式
    metric = Accumulator(2)  # 正確預測數、預測總數
    with torch.no_grad():
        X: torch.Tensor
        y: torch.Tensor
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            metric.add(count_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                dataloader: data.DataLoader,
                loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                trainer: Union[Callable[[float, int], None], torch.optim.Optimizer],
                device="cpu"):
    """訓練模型一個迭代週期"""
    # 將模型設定為訓練模式
    if isinstance(net, nn.Module):
        net.train()
    # 訓練損失總和、訓練精準度總和、樣本數
    metric = Accumulator(3)
    X: torch.Tensor
    y: torch.Tensor
    for X, y in dataloader:
        # 計算梯度並更新參數
        X = X.to(device)
        y = y.to(device)

        y_hat = net(X)
        lo = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer):
            # 使用PyTorch內建的最佳化器和損失函數
            trainer.zero_grad()
            lo.mean().backward()
            trainer.step()
        else:
            # 使用定製的最佳化器和損失函數
            lo.sum().backward()
            trainer(X.shape[0])
        metric.add(float(lo.sum()), count_accuracy(y_hat, y), y.numel())
    # 返回訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ani(net: nn.Module,
              train_dataloader: data.DataLoader,
              test_dataloader: data.DataLoader,
              loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              num_epochs: int,
              trainer: torch.optim.Optimizer,
              device="cpu"):
    """訓練模型"""
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_dataloader, loss, trainer, device)
        test_acc = evaluate_accuracy(net, test_dataloader, device)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
        train_loss, train_acc = train_metrics
        print(f'Train loss in {epoch} = {train_loss:.3f}')
        print(f'Train accuarcy in {epoch} = {train_acc:.3f}')
        print(f'Test accuarcy in {epoch} = {test_acc:.3f}')

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def set_axes(axes: Axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


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
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
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


def show_images(imgs: torch.Tensor, num_rows: int, num_cols: int, titles: tuple[str] = None, scale=1.5):
    """繪製圖像列表"""
    figsize = (num_cols * scale, (num_rows + 1) * scale)
    axes_arr: numpy.ndarray[Axes]
    _, axes_arr = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes_arr = axes_arr.flatten()
    axes: Axes

    for axes in axes_arr:
        axes.axes.get_xaxis().set_visible(False)
        axes.axes.get_yaxis().set_visible(False)

    for i, (axes, img) in enumerate(zip(axes_arr, imgs)):
        if torch.is_tensor(img):
            # to cpu to use numpy()
            pixels = img.to("cpu").numpy()  # (28, 28)
            axes.imshow(pixels)
        else:
            # PIL圖片
            axes.imshow(img)
        if titles:
            axes.set_title(titles[i])
    return axes_arr


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    plt.rcParams['figure.figsize'] = figsize


def plot(X: torch.Tensor,
         Y: torch.Tensor = None,
         xlabel=None,
         ylabel=None,
         legend=[],
         xlim=None,
         ylim=None,
         xscale='linear',
         yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5),
         axes: Axes = None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    def has_one_axis(X: torch.Tensor):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)

    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

    


def predict(net: nn.Module, test_dataloader: data.DataLoader, number=10, device="cpu"):
    """預測標籤"""
    X: torch.Tensor
    y: torch.Tensor
    X, y = next(iter(test_dataloader))
    X = X.to(device)
    y = y.to(device)

    labels = get_fashion_mnist_labels(y)
    t: torch.Tensor = net(X)

    preds = get_fashion_mnist_labels(t.argmax(axis=1))
    titles = [label + '\n' + pred for label, pred in zip(labels, preds)]
    num_cols = min(10, number)
    num_rows = math.ceil(number / num_cols)
    show_images(X[0:number].reshape((number, 28, 28)), num_rows, num_cols, titles=titles[0:number])


def savefig(path: str):
    plt.savefig(path)


def try_gpu(i=0):  #@save
    """如果存在,則返回gpu(i),否則返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU,如果沒有GPU,則返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]