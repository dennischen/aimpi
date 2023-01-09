import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
y = net(X)
print(f'{y}')

print('1===================================')
print(net.state_dict())
print(net[0].state_dict())
print(net[1].state_dict())
print(net[2].state_dict())

print('2===================================')
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print('3===================================')
print(f'{net[2].weight.grad == None}')

print('4===================================')
for name, param in net[0].named_parameters():
    print(f'{name}> {param}')

print('5===================================')
for name, param in net.named_parameters():
    print(f'{name}> {param}')

print('6===================================')
print(f'''{net.state_dict()['0.bias'].data}''')
print(f'''{net.state_dict()['2.bias'].data}''')

print('7===================================')


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在這裡巢狀
        net.add_module(f'block_{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
y = rgnet(X)

print(y)
print(rgnet.state_dict())
print(rgnet)

print('8===================================')
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def init_normal(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(f'{net[0].weight.data[0]}')
print(f'{net[0].bias.data[0]}')

print('9===================================')


def my_init(m: nn.Module):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape, param.data) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(f'{net[0].weight[:2]}')

