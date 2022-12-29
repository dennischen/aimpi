import random
import torch
from d2l import torch as d2l

from utils import basename_noext, kmp_duplicate_lib_ok

kmp_duplicate_lib_ok()

DEBUG = False
# generate training data from a source w,b
num_examples = 1000
source_weight = torch.tensor([2, -3.4])
source_bias = 4.2

print(f'>> Source weight: {source_weight}')
print(f'>> Source bias: {source_bias}')


def generate_simuliation_data(src_w: torch.Tensor, src_b: float, num: int):
    """生成y=Xw+b+噪聲"""

    if (DEBUG): print(f'src_w = {src_w}')
    if (DEBUG): print(f'src_b = {src_b}')

    size = (num, len(src_w))
    if (DEBUG): print(f'data size = {size}')

    # normal(mean:float, std:float, size:tuple())
    X = torch.normal(0, 1, size)
    if (DEBUG): print(f'X = {X}')

    Xw = torch.matmul(X, src_w)
    if (DEBUG): print(f'Xw = {Xw}')
    # equals to
    # print(f'Xw:Step1 = {X * w}')
    # print(f'Xw:Step2 = {torch.sum(X * w, dim=1)}')

    # Xw+b
    y: torch.Tensor = Xw + src_b
    if (DEBUG): print(f'y = Xw + b = {y}')

    # 噪聲
    n = torch.normal(0, 0.01, y.shape)
    if (DEBUG): print(f'n = {n}')

    yn = y + n

    if (DEBUG): print(f'yn = {yn}')

    yn = yn.reshape(-1, 1)
    if (DEBUG): print(f'yn.reshape = {yn}')

    return X, yn


# generate the X and y for training simulation from source
features, labels = generate_simuliation_data(source_weight, source_bias, num_examples)

if (DEBUG): print(f'>> Generated training data: \n>> Features {features}\n>> Labels:{labels}')

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.savefig(f'out/{basename_noext(__file__)}.png')


def random_shuffle_iter_data(batch_size, features, labels):
    num = len(features)
    indices = list(range(num))
    # 這些樣本是隨機讀取的，沒有特定的順序
    random.shuffle(indices)
    if (DEBUG): print(f'indices {indices}')
    # range(from, to, step)
    for i in range(0, num, batch_size):
        end = min(i + batch_size, num)
        if (DEBUG): print(f'Iterate [{i}:{end}] get {indices[i: end]}')
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num)])
        yield features[batch_indices], labels[batch_indices]


# here is batch concept demo
batch_size = 10
# random features, labels to a batch iteration
iter_data = random_shuffle_iter_data(batch_size, features, labels)
for idx, (X, y) in enumerate(iter_data):
    if (DEBUG):
        print(f'Batch {idx}: {X}, {y}')


# Start training
# define training method
def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """線性回歸模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """均方損失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params: tuple[torch.Tensor], lr: float, batch_size: int):
    """小批次隨機梯度下降"""
    with torch.no_grad():
        for param in params:
            # must use same tensor
            param[:] = param - lr * param.grad / batch_size
            # or use -=
            # param -= lr * param.grad / batch_size
            param.grad.zero_()


# the initial value of a training w, b
weight = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

print(f'>> Initialized weight: {weight}')
print(f'>> Initialized bias: {bias}')

batch_size = 10
lr = 0.03
num_epochs = 3
network = linreg
loss = squared_loss
optimize = sgd

with torch.no_grad():
    train_loss = loss(network(features, weight, bias), labels)
    print(f'>> Initial loss {float(train_loss.mean()):f}')

print(f'>> Initial difference weight: {source_weight - weight.reshape(source_weight.shape)}')
print(f'>> Initial difference bias: {source_bias - bias}')

for epoch in range(num_epochs):
    for X, y in random_shuffle_iter_data(batch_size, features, labels):
        batch_loss = loss(network(X, weight, bias), y)  # X和y的小批次損失
        # 因為l形狀是(batch_size,1)，而不是一個標量。
        # l中的所有元素被加到一起，
        # 並以此計算關於[w,b]的梯度

        # backpropagation
        batch_loss.sum().backward()
        optimize([weight, bias], lr, batch_size)  # 使用參數的梯度更新參數
    with torch.no_grad():
        train_loss = loss(network(features, weight, bias), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

print(f'>> weight difference from source: {source_weight - weight.reshape(source_weight.shape)}')
print(f'>> bias difference from source: {source_bias - bias}')
