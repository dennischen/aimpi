import time
import math
import os
from typing import Callable, Union
import numpy as np

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


def load_data_fashion_mnist(batch_size: int, resize: int = None):

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="download-data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="download-data", train=False, transform=trans, download=True)

    train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def get_fashion_mnist_labels(labels):
    """??????Fashion-MNIST????????????????????????"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data(data_arrays: torch.Tensor, batch_size: int, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


class Accumulator:
    """???n??????????????????"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def count_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """???????????????????????????"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                      dataloader: data.DataLoader,
                      device="cpu"):
    """??????????????????????????????????????????"""
    if isinstance(net, nn.Module):
        net.eval()  # ??????????????????????????????
    metric = Accumulator(2)  # ??????????????????????????????
    with torch.no_grad():
        X: torch.Tensor
        y: torch.Tensor
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            metric.add(count_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_gpu(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                          data_loader: data.DataLoader,
                          device: torch.device = None):
    """??????GPU????????????????????????????????????"""
    if isinstance(net, nn.Module):
        net.eval()  # ?????????????????????
        if not device:
            device = next(iter(net.parameters())).device
    # ?????????????????????,??????????????????
    metric = Accumulator(2)
    with torch.no_grad():
        X: torch.Tensor
        y: torch.Tensor
        for X, y in data_loader:
            if isinstance(X, list):
                # BERT????????????????????????????????????
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(count_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
                dataloader: data.DataLoader,
                loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                trainer: Union[Callable[[float, int], None], torch.optim.Optimizer],
                device="cpu"):
    """??????????????????????????????"""
    # ??????????????????????????????
    if isinstance(net, nn.Module):
        net.train()
    # ??????????????????????????????????????????????????????
    metric = Accumulator(3)
    X: torch.Tensor
    y: torch.Tensor
    for X, y in dataloader:
        # ???????????????????????????
        X = X.to(device)
        y = y.to(device)

        y_hat = net(X)
        lo = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer):
            # ??????PyTorch????????????????????????????????????
            trainer.zero_grad()
            lo.mean().backward()
            trainer.step()
        else:
            # ??????????????????????????????????????????
            lo.sum().backward()
            trainer(X.shape[0])
        metric.add(float(lo.sum()), count_accuracy(y_hat, y), y.numel())
    # ?????????????????????????????????
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ani(net: nn.Module,
              train_dataloader: data.DataLoader,
              test_dataloader: data.DataLoader,
              loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              num_epochs: int,
              trainer: torch.optim.Optimizer,
              device="cpu"):
    """????????????"""
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
    """????????????????????????"""
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
        # ????????????????????????
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # ??????lambda??????????????????
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.ion()
        self.fig.show()

    def add(self, x, y):
        # ?????????????????????????????????
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
    """??????????????????"""
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
            # PIL??????
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
    """????????????"""
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
    """????????????,?????????gpu(i),????????????cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """?????????????????????GPU,????????????GPU,?????????[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def train_ch6(net: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module], train_dataloader: data.DataLoader,
              test_dataloader: data.DataLoader, num_epochs: int, lr: float, device: torch.device):
    """???GPU????????????(??????????????????)"""
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
        timer_epoch = Timer()
        timer_epoch.start()
        print(f'start epoch {epoch}, batches {num_batches}')
        # ??????????????????,?????????????????????,?????????
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
            train_lo = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_lo, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_dataloader)
        animator.add(epoch + 1, (None, None, test_acc))
        timer_epoch.stop()
        print(f'epoch {epoch} tooks {timer_epoch.sum()} sec for {metric[2]} examples')
    print(f'loss {train_lo:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')