import random
import torch
from d2l import torch as d2l

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DEBUG = False


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪聲"""

    if(DEBUG):
        print(f'w = {w}')
    if(DEBUG):
        print(f'b = {b}')

    size = (num_examples, len(w))
    if(DEBUG):
        print(f'size = {size}')

    # normal(mean:float, std:float, size:tuple())
    X = torch.normal(0, 1, size)
    if(DEBUG):
        print(f'X = {X}')

    Xw = torch.matmul(X, w)
    if(DEBUG):
        print(f'Xw = {Xw}')
    # equals to
    # print(f'Xw:Step1 = {X * w}')
    # print(f'Xw:Step2 = {torch.sum(X * w, dim=1)}')

    # Xw+b
    y: torch.Tensor = Xw + b
    if(DEBUG):
        print(f'y = Xw + b = {y}')

    # 噪聲
    n = torch.normal(0, 0.01, y.shape)
    if(DEBUG):
        print(f'n = {n}')

    yn = y + n

    if(DEBUG):
        print(f'yn = {yn}')
    if(DEBUG):
        print(f'yn.reshape = {yn.reshape(-1, 1)}')

    return X, yn.reshape((-1, 1))


# generate training data from a assigned w,b
num_examples = 1000
true_weight = torch.tensor([2, -3.4])
true_bais = 4.2
print(f'>> True Weight:{true_weight}')
print(f'>> True Bais:{true_bais}')

features, labels = synthetic_data(true_weight, true_bais, num_examples)

print(
    f'>> Generated training data: \n>> Features {features}\n>> Labels:{labels}')

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.savefig(f'out/{stem(__file__)}.png')


def data_random_shuffle_iter(batch_size, features, labels):
    num = len(features)
    indices = list(range(num))
    # 這些樣本是隨機讀取的，沒有特定的順序
    random.shuffle(indices)
    if(DEBUG):
        print(f'indices {indices}')
    # range(from, to, step)
    for i in range(0, num, batch_size):
        end = min(i + batch_size, num)
        if(DEBUG):
            print(f'Iterate [{i}:{end}] get {indices[i: end]}')
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num)])
        yield features[batch_indices], labels[batch_indices]


# demo batch concept
batch_size = 2
# random features, labels to a batch iteration
iter = data_random_shuffle_iter(batch_size, features, labels)
for idx, (X, y) in enumerate(iter):
    if(DEBUG):
        print(f'Batch {idx}: {X}, {y}')


# the initial value of a training w, b
weight = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
bais = torch.zeros(1, requires_grad=True)

print(f'Initialized Weight:{weight}')
print(f'Initialized Bais:{bais}')


def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """線性回歸模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat:torch.Tensor, y:torch.Tensor):
    """均方損失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params: tuple[torch.Tensor], lr: float, batch_size: int):
    """小批次隨機梯度下降"""
    with torch.no_grad():
        for param in params:
            # must use same tensor
            param[:] = param -  lr * param.grad / batch_size
            # or use -=
            # param -= lr * param.grad / batch_size
            param.grad.zero_()


# Start training
batch_size = 10
lr = 0.03
num_epochs = 3
network = linreg
loss = squared_loss
optimize = sgd

for epoch in range(num_epochs):
    for X, y in data_random_shuffle_iter(batch_size, features, labels):
        batch_loss = loss(network(X, weight, bais), y)  # X和y的小批次損失
        # 因為l形狀是(batch_size,1)，而不是一個標量。
        # l中的所有元素被加到一起，
        # 並以此計算關於[w,b]的梯度

        # backpropagation
        batch_loss.sum().backward()
        optimize([weight, bais], lr, batch_size)  # 使用參數的梯度更新參數
    with torch.no_grad():
        train_loss = loss(network(features, weight, bais), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')


print(f'weight的估計誤差: {true_weight - weight.reshape(true_weight.shape)}')
print(f'bais的估計誤差: {true_bais - bais}')
