import torch
from utils import (basename_noext)
from torch import nn
from torch.nn import functional as F

x = torch.randn(4, 5)
print(x)
torch.save(x, f'out/{basename_noext(__file__)}.zip')

x1 = torch.load(f'out/{basename_noext(__file__)}.zip')
print(x1)

assert torch.all(x == x1)

y = torch.zeros(4)
torch.save([x, y], f'{basename_noext(__file__)}.zip')
x2, y2 = torch.load(f'{basename_noext(__file__)}.zip')

print(x2)
print(y2)
assert torch.all(x == x2)
assert torch.all(y == y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, f'{basename_noext(__file__)}.zip')
mydict2 = torch.load(f'{basename_noext(__file__)}.zip')
print(mydict2)



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
print(f'MLP {net}')
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), f'out/{basename_noext(__file__)}_mlp.params')

clone = MLP()
clone.load_state_dict(torch.load(f'out/{basename_noext(__file__)}_mlp.params'))
clone.eval()

Y1 = net(X)
Y2 = clone(X)

print(f'{torch.all(Y1 == Y2)}')

assert torch.all(Y1 == Y2)
