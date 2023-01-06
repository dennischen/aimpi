import torch
from d2l import torch as d2l
from utils import (basename_noext, get_fashion_mnist_labels, kmp_duplicate_lib_ok, load_data_fashion_minist, train,
                   predict, savefig)

kmp_duplicate_lib_ok()
torch.set_printoptions(linewidth=120)


####################
# -8 to 8 , step 0.1
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
print(f'x {x}')
d2l.plot(x.detach(), x.detach(), 'x', 'x', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}.png')

####################
# relu(x) = max(x, 0)
y = torch.relu(x)
print(f'relu {y}')
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_relu.png')

y.backward(torch.ones_like(x), retain_graph=True)
print(f'relu grad {x.grad}')

d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_relu_grad.png')


##################
# sigmoid(x) = 1 / (1 + exp(-x))
z = torch.sigmoid(x)
print(f'sigmoid {z}')
d2l.plot(x.detach(), z.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_sigmoid.png')

x.grad.zero_()
z.backward(torch.ones_like(x), retain_graph=True)
print(f'sigmoid grad {x.grad}')

d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_sigmoid_grad.png')


##################
# tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
t = torch.tanh(x)
print(f'sigmoid {t}')
d2l.plot(x.detach(), t.detach(), 'x', 'tanh(t)', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_tanh.png')

x.grad.zero_()
t.backward(torch.ones_like(x), retain_graph=True)
print(f'sigmoid tanh {x.grad}')

d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
savefig(f'out/{basename_noext(__file__)}_tanh_grad.png')