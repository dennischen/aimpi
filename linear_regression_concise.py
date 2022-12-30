import torch
from torch.utils import data
from torch import nn

from d2l import torch as d2l

from utils import basename_noext, kmp_duplicate_lib_ok

kmp_duplicate_lib_ok()

DEBUG = False
num_examples = 1000
source_weight = torch.tensor([2, -3.4])
source_bias = 4.2

print(f'>> Source weight: {source_weight}')
print(f'>> Source bias: {source_bias}')

features: torch.Tensor
labels: torch.Tensor
features, labels = d2l.synthetic_data(source_weight, source_bias, num_examples)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.savefig(f'out/{basename_noext(__file__)}.png')


def data_loader(data_arrays, batch_size, is_train=True):
    """構造一個PyTorch資料迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
lr = 0.03
num_epochs = 3
iter_data = data_loader((features, labels), batch_size)

# define a 2 inputs > 1 output network
l1 = nn.Linear(2, 1)
# inintal layer

l1.weight.data.normal_(0, 0.01)
l1.bias.data.fill_(0)

print(f'>> Initialized weight: {l1.weight.data}')
print(f'>> Initialized bias: {l1.bias.data}')

# chain layers
net = nn.Sequential(l1)

assert l1 == net[0]


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


loss = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr)

with torch.no_grad():
    initial_loss = loss(net(features), labels)
    print(f'initial loss {initial_loss:f}')
w = l1.weight.data
print(f'initial different weight: {source_weight - w.reshape(source_weight.shape)}')
b = l1.bias.data
print(f'initial different bias: {source_bias - b}')

for epoch in range(num_epochs):
    for X, y in iter_data:
        # ==the scratch way==
        # batch_loss = squared_loss(net(X), y)
        # batch_loss.sum().backward()
        # sgd(tuple(i for i in net.parameters()), lr, batch_size)

        # ==the mixed==
        # optim.zero_grad()
        # batch_loss = squared_loss(net(X), y)
        # batch_loss.sum().backward()
        # optim.step()

        # ==the concise way==
        optim.zero_grad()
        batch_loss: torch.Tensor = loss(net(X), y)
        # l = loss.forward(net.forward(X) ,y) # net.forward(X) is same as net(X)
        batch_loss.backward()
        optim.step()
    with torch.no_grad():
        # ==the scratch way==
        # train_loss = squared_loss(net(features), labels)
        # print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

        # ==the mixed==
        # train_loss = squared_loss(net(features), labels)
        # print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

        # ==the concise way==
        train_loss = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {train_loss:f}')

w = l1.weight.data
print(f'different in estimating weight: {source_weight - w.reshape(source_weight.shape)}')
b = l1.bias.data
print(f'different in estimating bias: {source_bias - b}')
